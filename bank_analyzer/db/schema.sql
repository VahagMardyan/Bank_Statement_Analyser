CREATE DATABASE IF NOT EXISTS bank_analyser
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE bank_analyser;

-- User
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    language VARCHAR(5) NOT NULL DEFAULT "hy",
    notification_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    remind_days_before TINYINT NOT NULL DEFAULT 3,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Transactions
CREATE TABLE transactions(
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    txn_date DATETIME NOT NULL,
    description TEXT NOT NULL,
    amount DECIMAL(14, 2) NOT NULL,          -- always AMD (converted if needed)
    original_amount DECIMAL(14, 2),          -- as written in the statement
    original_currency CHAR(3) NOT NULL DEFAULT 'AMD',
    transaction_type ENUM('income', 'expense') NOT NULL,
    category VARCHAR(64) NOT NULL,
    method VARCHAR(32),
    confidence DECIMAL(4, 3),
    fingerprint CHAR(64) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_user_txn (user_id, fingerprint),
    KEY ix_user_date (user_id, txn_date),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- User manual rules
CREATE TABLE user_rules(
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    category VARCHAR(64) NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    UNIQUE KEY uq_user_keyword (user_id, keyword),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ML Rules
CREATE TABLE training_examples (
    id        INT PRIMARY KEY AUTO_INCREMENT,
    user_id   INT          NOT NULL,
    text      TEXT         NOT NULL,
    category  VARCHAR(64)  NOT NULL,
    source    VARCHAR(16)  NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Budgets
CREATE TABLE budgets (
    user_id INT NOT NULL,
    category VARCHAR(64) NOT NULL,
    monthly_limit DECIMAL(14, 2) NOT NULL,
    PRIMARY KEY (user_id, category),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Recurring payments
CREATE TABLE recurring_payments (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    user_id        INT           NOT NULL,
    name           VARCHAR(255)  NOT NULL,
    category       VARCHAR(64)   NOT NULL,
    expected_amount DECIMAL(14,2),
    interval_days  SMALLINT      NOT NULL,
    last_paid      DATE,
    next_due       DATE          NOT NULL,
    source         ENUM('detected','manual') NOT NULL,
    active         BOOLEAN       NOT NULL DEFAULT TRUE,
    dismissed      BOOLEAN       NOT NULL DEFAULT FALSE,
    UNIQUE KEY uq_user_name (user_id, name),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 7. Already sent notifications
CREATE TABLE notifications_log (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    user_id       INT          NOT NULL,
    kind          ENUM('reminder','advice') NOT NULL,
    reference_key VARCHAR(100) NOT NULL,
    sent_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_once (user_id, kind, reference_key),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);


-- 8. "Keep me signed in" tokens (the cookie holds selector:validator, never the user id)
CREATE TABLE auth_tokens (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT       NOT NULL,
    selector    CHAR(24)  NOT NULL UNIQUE,
    token_hash  CHAR(64)  NOT NULL,
    expires_at  DATETIME  NOT NULL,
    created_at  DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
