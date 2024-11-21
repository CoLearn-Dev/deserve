use anyhow::Result;
use bytes::Bytes;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyDict, PyList};
use pyo3::Bound as PyBound;
use safetensors::{Dtype, SafeTensors, View};

use std::borrow::Cow;
use std::collections::HashMap;

pub struct PyView {
    shape: Vec<usize>,
    dtype: Dtype,
    data: Bytes,
    data_len: usize,
}

impl PyView {
    pub fn new(shape: Vec<usize>, dtype: Dtype, data: Bytes, data_len: usize) -> Self {
        Self {
            shape,
            dtype,
            data,
            data_len,
        }
    }

    pub fn into_py(self, py: Python) -> HashMap<String, PyObject> {
        let shape: PyObject = PyList::new_bound(py, self.shape.iter()).into();
        let dtype: PyObject = format!("{:?}", self.dtype).into_py(py);
        let data: PyObject = PyByteArray::new_bound(py, self.data.as_ref()).into();
        HashMap::from([
            ("shape".to_string(), shape),
            ("dtype".to_string(), dtype),
            ("data".to_string(), data),
        ])
    }
}

impl View for &PyView {
    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(&self.data)
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn data_len(&self) -> usize {
        self.data_len
    }
}

pub fn prepare(tensor_dict: HashMap<String, PyBound<PyDict>>) -> PyResult<HashMap<String, PyView>> {
    let mut tensors = HashMap::with_capacity(tensor_dict.len());
    for (tensor_name, tensor_desc) in &tensor_dict {
        let shape: Vec<usize> = tensor_desc
            .get_item("shape")?
            .ok_or_else(|| SafetensorError::new_err(format!("Missing `shape` in {tensor_desc:?}")))?
            .extract()?;
        let pydata: PyBound<PyAny> = tensor_desc.get_item("data")?.ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `data` in {tensor_desc:?}"))
        })?;
        // Make sure it's extractable first.
        let data: &[u8] = pydata.extract()?;
        let data_len = data.len();
        let data: PyBound<PyBytes> = pydata.extract()?;
        let data = Bytes::copy_from_slice(data.as_bytes()); // use copy for doing in background
        let pydtype = tensor_desc.get_item("dtype")?.ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `dtype` in {tensor_desc:?}"))
        })?;
        let dtype: String = pydtype.extract()?;
        let dtype = match dtype.as_ref() {
            "bool" => Dtype::BOOL,
            "int8" => Dtype::I8,
            "uint8" => Dtype::U8,
            "int16" => Dtype::I16,
            "uint16" => Dtype::U16,
            "int32" => Dtype::I32,
            "uint32" => Dtype::U32,
            "int64" => Dtype::I64,
            "uint64" => Dtype::U64,
            "float16" => Dtype::F16,
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            "bfloat16" => Dtype::BF16,
            "float8_e4m3fn" => Dtype::F8_E4M3,
            "float8_e5m2" => Dtype::F8_E5M2,
            dtype_str => {
                return Err(SafetensorError::new_err(format!(
                    "dtype {dtype_str} is not covered",
                )));
            }
        };

        let tensor = PyView {
            shape,
            dtype,
            data,
            data_len,
        };
        tensors.insert(tensor_name.to_string(), tensor);
    }
    Ok(tensors)
}

pub fn serialize(tensors: HashMap<String, PyView>) -> Result<Bytes> {
    Ok(Bytes::from(
        safetensors::tensor::serialize(&tensors, &None)
            .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e:?}")))?,
    ))
}

// pub fn deserialize(bytes: &[u8]) -> Result<SafeTensors> {
//     SafeTensors::deserialize(bytes).map_err(|e| anyhow::anyhow!("Error while deserializing: {e:?}"))
// }

// pub fn convert_tensors_to_py(
//     py: Python,
//     safetensors: SafeTensors,
// ) -> PyResult<Vec<(String, HashMap<String, PyObject>)>> {
//     let tensors = safetensors.tensors();
//     let mut items = Vec::with_capacity(tensors.len());

//     for (tensor_name, tensor) in tensors {
//         let pyshape: PyObject = PyList::new_bound(py, tensor.shape().iter()).into();
//         let pydtype: PyObject = format!("{:?}", tensor.dtype()).into_py(py);

//         let pydata: PyObject = PyByteArray::new_bound(py, tensor.data()).into();

//         let map = HashMap::from([
//             ("shape".to_string(), pyshape),
//             ("dtype".to_string(), pydtype),
//             ("data".to_string(), pydata),
//         ]);
//         items.push((tensor_name, map));
//     }
//     Ok(items)
// }

pub fn deserialize(bytes: &[u8]) -> PyResult<Vec<(String, PyView)>> {
    let safetensor = SafeTensors::deserialize(bytes)
        .map_err(|e| SafetensorError::new_err(format!("Error while deserializing: {e:?}")))?;

    let tensors = safetensor.tensors();
    let mut items = Vec::with_capacity(tensors.len());

    for (tensor_name, tensor) in tensors {
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();
        let data = Bytes::copy_from_slice(tensor.data());
        let data_len = data.len();

        items.push((tensor_name, PyView::new(shape, dtype, data, data_len)));
    }
    Ok(items)
}

pyo3::create_exception!(
    safetensors_rust,
    SafetensorError,
    PyException,
    "Custom Python Exception for Safetensor errors."
);
