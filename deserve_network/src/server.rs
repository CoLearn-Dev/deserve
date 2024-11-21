use std::{collections::HashMap, sync::Arc};

use actix_web::{
    web::{self, Data},
    App, HttpRequest, HttpResponse, HttpServer,
};
use bytes::{self, Bytes, BytesMut};
use env_logger;
use pyo3::{
    exceptions::PyException,
    pyclass, pymethods,
    types::{PyByteArray, PyBytes, PyBytesMethods},
    Bound as PyBound, PyErr, PyObject, Python,
};
use pyo3::{types::PyDict, Py, PyResult};
use reqwest::Client;
use tokio::sync::mpsc::{self, Sender};

use crate::sede::{deserialize, prepare, serialize, PyView};

#[pyclass]
pub struct PyServer {
    runtime: Arc<tokio::runtime::Runtime>,
    receiver: Option<mpsc::Receiver<(String, bytes::Bytes)>>,
    client: Client,
}

#[pymethods]
impl PyServer {
    #[new]
    fn new(
        address: &str,
        routes: Vec<String>,
        worker_threads: usize,
        enable_logging: bool,
    ) -> Self {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(worker_threads)
                .enable_all()
                .build()
                .unwrap(),
        );
        let (tx, rx) = mpsc::channel::<(String, bytes::Bytes)>(128);
        let address = address.to_string();

        if enable_logging {
            std::env::set_var("RUST_LOG", "debug");
            env_logger::init();
        }

        let routes0 = routes.clone();
        let data_tx = Data::new(tx.clone());
        runtime.spawn(async {
            println!("listening on {}", address);
            HttpServer::new(move || {
                let mut app = App::new()
                    .app_data(data_tx.clone())
                    .app_data(web::PayloadConfig::new(16 * 1024 * 1024));
                for route in &routes0 {
                    app = app.route(route.as_str(), web::post().to(step));
                }
                app
            })
            .bind(address)
            .unwrap()
            .run()
            .await
            .unwrap();
        });

        Self {
            runtime,
            receiver: Some(rx),
            client: Client::new(),
        }
    }

    fn recv_tensors(
        this: Py<Self>,
        py: Python<'_>,
    ) -> PyResult<(
        String,
        Vec<(String, HashMap<String, PyObject>)>,
        Py<PyByteArray>,
    )> {
        let mut receiver = this.borrow_mut(py).receiver.take().unwrap();
        let (receiver, result) = Python::allow_threads(py, move || {
            let result = receiver
                .blocking_recv()
                .ok_or_else(|| DeserveNetworkError::new_err("No message received"));
            let (route, body) = match result {
                Err(e) => return (receiver, Err(e)),
                Ok((route, body)) => (route, body),
            };

            let len_tensors = u32::from_be_bytes(body[..4].try_into().unwrap());
            let tensors = match deserialize(&body[4..(len_tensors as usize + 4)]) {
                Ok(t) => t,
                Err(e) => {
                    return (
                        receiver,
                        Err(DeserveNetworkError::new_err(format!(
                            "Error while deserializing: {e:?}"
                        ))),
                    )
                }
            };
            (
                receiver,
                Ok((
                    route,
                    tensors,
                    Bytes::copy_from_slice(&body[(len_tensors as usize + 4)..]),
                )),
            )
        });
        this.borrow_mut(py).receiver = Some(receiver);

        match result {
            Ok((route, tensors, metadata)) => Ok((
                route,
                tensors
                    .into_iter()
                    .map(|(name, tensor)| (name.clone(), tensor.into_py(py)))
                    .collect(),
                PyByteArray::new_bound(py, metadata.as_ref()).into(),
            )),
            Err(e) => Err(e),
        }
    }

    fn send_tensors(
        this: Py<Self>,
        py: Python<'_>,
        address: &str,
        tensor_dict: HashMap<String, PyBound<PyDict>>,
        metadata: PyBound<PyBytes>,
    ) -> PyResult<bool> {
        let runtime = this.borrow(py).runtime.clone();
        let tensors = prepare(tensor_dict)?;
        let client = this.borrow(py).client.clone();
        let metadata = Bytes::from(metadata.as_borrowed().as_bytes().to_vec());
        Python::allow_threads(py, move || {
            send_serialized_tensors(runtime, client, address, tensors, metadata)
        })
    }
}

#[pyclass]
pub struct PyClient {
    runtime: Arc<tokio::runtime::Runtime>,
    client: Client,
}

#[pymethods]
impl PyClient {
    #[new]
    fn new(worker_threads: usize) -> Self {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(worker_threads)
                .enable_all()
                .build()
                .unwrap(),
        );
        let client = Client::new();
        Self { runtime, client }
    }

    fn send_tensors(
        this: Py<Self>,
        py: Python<'_>,
        address: &str,
        tensor_dict: HashMap<String, PyBound<PyDict>>,
        metadata: PyBound<PyBytes>,
    ) -> PyResult<bool> {
        let runtime = this.borrow(py).runtime.clone();
        let tensors = prepare(tensor_dict)?;
        let client = this.borrow(py).client.clone();
        let metadata = Bytes::from(metadata.as_borrowed().as_bytes().to_vec());
        Python::allow_threads(py, move || {
            send_serialized_tensors(runtime, client, address, tensors, metadata)
        })
    }
}

#[inline]
fn send_serialized_tensors(
    runtime: Arc<tokio::runtime::Runtime>,
    client: Client,
    address: &str,
    tensors: HashMap<String, PyView>,
    metadata: Bytes,
) -> Result<bool, PyErr> {
    runtime.block_on(async {
        let mut serialized = BytesMut::new();
        let serialized_tensors = serialize(tensors).unwrap();
        serialized.extend_from_slice(&(serialized_tensors.len() as u32).to_be_bytes());
        serialized.extend_from_slice(&serialized_tensors);
        serialized.extend_from_slice(&metadata);
        client
            .post(address)
            .header("Content-Type", "application/octet-stream")
            .body(serialized.freeze())
            .send()
            .await
            .map(|res| {
                if !res.status().is_success() {
                    println!("{:?}", res.status());
                    false
                } else {
                    res.status().is_success()
                }
            })
            .map_err(|e| DeserveNetworkError::new_err(format!("Failed to send request: {:?}", e)))
    })
}

async fn step(
    tx: Data<Sender<(String, bytes::Bytes)>>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    let route = request.uri().path();
    tx.send((route.to_string(), body)).await.unwrap();
    HttpResponse::Ok().body("ok")
}

pyo3::create_exception!(
    deserve_network_rust,
    DeserveNetworkError,
    PyException,
    "Custom Python Exception for Deserve network errors."
);
