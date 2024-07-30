-- Adminer 4.8.1 MySQL 8.0.36 dump

SET NAMES utf8;
SET time_zone = '+00:00';
SET foreign_key_checks = 0;
SET sql_mode = 'NO_AUTO_VALUE_ON_ZERO';

SET NAMES utf8mb4;

DROP TABLE IF EXISTS `chat_history`;
CREATE TABLE `chat_history` (
  `id` text NOT NULL,
  `email` text NOT NULL,
  `status` text NOT NULL,
  `timestamp` timestamp NOT NULL,
  `req_body` text NOT NULL,
  `plans` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


DROP TABLE IF EXISTS `credit_transaction`;
CREATE TABLE `credit_transaction` (
  `email` text NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `event_type` text NOT NULL,
  `amount` bigint NOT NULL,
  `event_detail` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


DROP TABLE IF EXISTS `network_server`;
CREATE TABLE `network_server` (
  `type` text NOT NULL,
  `url` text NOT NULL,
  `username` text NOT NULL,
  `password` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


DROP TABLE IF EXISTS `perf_computation`;
CREATE TABLE `perf_computation` (
  `worker_id` text NOT NULL,
  `layers` text NOT NULL,
  `input_shape` text NOT NULL,
  `latency` float NOT NULL,
  UNIQUE KEY `worker_id_layers_input_shape` (`worker_id`(36),`layers`(100),`input_shape`(32))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


DROP TABLE IF EXISTS `perf_network`;
CREATE TABLE `perf_network` (
  `from_worker_id` text NOT NULL,
  `to_worker_id` text NOT NULL,
  `latency` float NOT NULL,
  UNIQUE KEY `from_worker_id_to_worker_id` (`from_worker_id`(36),`to_worker_id`(36))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


DROP TABLE IF EXISTS `task_manager`;
CREATE TABLE `task_manager` (
  `name` text NOT NULL,
  `url` text NOT NULL,
  `pubkey` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


DROP TABLE IF EXISTS `user_api_token`;
CREATE TABLE `user_api_token` (
  `email` text NOT NULL,
  `token` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


DROP TABLE IF EXISTS `worker`;
CREATE TABLE `worker` (
  `id` text NOT NULL,
  `owner_email` text NOT NULL,
  `url` text NOT NULL,
  `version` text NOT NULL,
  `nickname` text NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_seen` timestamp NOT NULL DEFAULT '1970-01-01 00:00:01',
  `gpu_model` text NOT NULL,
  `gpu_total_memory` bigint NOT NULL,
  `gpu_remaining_memory` bigint NOT NULL,
  `loaded_layers` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


-- 2024-04-11 01:47:10