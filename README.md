# Blockchain-Python

![Blockchain](https://img.shields.io/badge/Blockchain-Python-blue)

This repository contains two implementations of a blockchain in Python:
1. **Basic Blockchain** - A simple blockchain demonstrating fundamental concepts.
2. **Advanced Blockchain** - An enhanced version with additional features and improvements.

## 🚀 Features
### Basic Blockchain
- Create a blockchain from scratch
- Implement proof-of-work (PoW) mechanism
- Validate transactions and chain integrity
- Simple Flask-based API for interaction
- Mine new blocks and add transactions

### Advanced Blockchain
- Smart contract support
- Improved consensus mechanism
- Enhanced security and validation
- Decentralized peer-to-peer communication

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhruvv1402/Blockchain-Python.git
   cd Blockchain-Python
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1️⃣ Running the Basic Blockchain
```bash
cd basic_blockchain
python basicchain.py
```
By default, it runs on `http://127.0.0.1:5000/`

### 2️⃣ Running the Advanced Blockchain
```bash
cd advanced_blockchain
python advancedchain.py
```

### 3️⃣ Available API Endpoints (Basic & Advanced)

- **Mine a new block**  
  ```
  GET /mine
  ```
  Mines a new block and adds it to the blockchain.

- **Get the full blockchain**  
  ```
  GET /chain
  ```
  Returns the entire blockchain in JSON format.

- **Add a new transaction**  
  ```
  POST /transactions/new
  ```
  Adds a new transaction to the next block.
  ```json
  {
      "sender": "Alice",
      "recipient": "Bob",
      "amount": 10
  }
  ```

## 🏗️ Project Structure
```
Blockchain-Python/
│-- basic_blockchain/       # Basic blockchain implementation
│   │-- basicchain.py       # Main basic blockchain logic
│-- advanced_blockchain/    # Advanced blockchain implementation
│   │-- advancedchain.py    # Advanced blockchain logic
│-- requirements.txt        # Required dependencies
│-- README.md               # Project documentation
```

## 🤝 Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to your branch (`git push origin feature-branch`)
5. Open a Pull Request

## 📜 License
This project is licensed under the MIT License.

---

Feel free to improve and expand this project! Happy coding! 🚀

