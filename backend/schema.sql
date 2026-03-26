CREATE DATABASE IF NOT EXISTS flavorsnap;
USE flavorsnap;

CREATE TABLE IF NOT EXISTS classification_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    label VARCHAR(255) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    image_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
