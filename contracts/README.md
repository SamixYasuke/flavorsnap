# ⛓️ FlavorSnap Smart Contracts

This directory contains the Soroban smart contracts that power the decentralized governance and incentive layers of FlavorSnap.

## 📋 Table of Contents

- [🌳 Overview](#-overview)
- [🏗️ Contract Architecture](#️-contract-architecture)
- [🛠️ Development Setup](#️-development-setup)
- [📦 Contract Modules](#-contract-modules)
  - [Model Governance](#model-governance)
  - [Tokenized Incentive](#tokenized-incentive)
  - [Sensory Evaluation](#sensory-evaluation)
- [🚀 Deployment](#-deployment)
- [🔒 Security Considerations](#-security-considerations)

## 🌳 Overview

FlavorSnap utilizes the Stellar network and Soroban smart contracts to create a transparent, trustless ecosystem for AI-powered food classification. The blockchain layer handles model versioning, community rewards, and sensory feedback validation.

## 🏗️ Contract Architecture

The system is composed of three primary smart contracts:

1.  **Model Governance**: Manages proposals for model updates and dataset expansions.
2.  **Tokenized Incentive**: Handles reward distribution and vesting for contributors.
3.  **Sensory Evaluation**: Manages community-driven feedback and staking for data validation.

## 🛠️ Development Setup

### Prerequisites

- **Rust**: [Install Rust](https://www.rust-lang.org/tools/install)
- **Soroban SDK**: Included in `Cargo.toml`
- **Stellar CLI**: `cargo install --locked stellar-cli`
- **Wasm Target**: `rustup target add wasm32-unknown-unknown`

### Building Contracts

To compile all contracts to WASM:

```bash
cargo build --target wasm32-unknown-unknown --release
```

### Running Tests

Each contract includes a comprehensive test suite. run them using:

```bash
cargo test
```

## 📦 Contract Modules

### Model Governance (`model-governance/`)

This contract ensures that only high-quality, community-approved models are used in production.

-   **Proposals**: Users can submit proposals for model updates with metadata links.
-   **Voting**: Token holders vote on proposals based on their stake.
-   **Quorum**: Configurable threshold for proposal approval.
-   **Execution**: Automated state updates upon successful voting rounds.

### Tokenized Incentive (`tokenized-incentive/`)

Aligns community interests with project growth through a transparent reward system.

-   **Mint/Burn**: Controlled token supply management.
-   **Vesting**: Multi-stage vesting schedules for long-term contributors.
-   **Multi-sig**: Critical admin actions (like minting) require multi-signature approval from defined admins.
-   **Action Approvals**: Internal tracking of admin approvals for sensitive operations.

### Sensory Evaluation (`sensory-evaluation/`)

Validates the "sensory" accuracy of food classifications through community staking.

-   **Staking**: Users stake tokens to vouch for the accuracy of a classification.
-   **Rewards**: Earn tokens for correct evaluations.
-   **Slashing**: Potential loss of stake for malicious or incorrect reporting (infrastructure-ready).
-   **Admin Management**: Secure management of evaluation parameters.

## 🚀 Deployment

Deployment is managed via the `scripts/deploy_contracts.sh` script.

```bash
# Deploy to Stellar Testnet
./scripts/deploy_contracts.sh testnet

# Deploy to Stellar Mainnet
./scripts/deploy_contracts.sh mainnet
```

Configurations for different networks are stored in `config/`.

## 🔒 Security Considerations

-   **Multi-Signature Admin**: Sensitive operations in `TokenizedIncentive` require multiple admin approvals.
-   **Role-Based Access**: Strict `admin` checks on all state-changing initialization and configuration functions.
-   **Staking Locks**: Tokens are locked during the evaluation period to prevent "nothing-at-stake" attacks.
-   **Data Sanitization**: Metadata strings are stored as Soroban `String` types to minimize on-chain footprint.

---

*Last updated: March 2026*
*For more detailed blockchain architecture, see [docs/blockchain.md](../docs/blockchain.md).*
