import hashlib
import json
import time
import threading
import random
import string
import base64
import os
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import requests
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature
from datetime import datetime
import logging
from flask import Flask, jsonify, request, render_template, redirect, url_for, session, flash
from flask_socketio import SocketIO, emit
from dataclasses import dataclass, asdict, field
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import io
import uuid
from cryptography.hazmat.primitives import hashes, padding, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.exceptions import InvalidSignature
import matplotlib.pyplot as plt
import networkx as nx
from flask_socketio import SocketIO, emit


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BlockchainApp')


@dataclass
class Transaction:
    sender: str
    recipient: str
    amount: float
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    transaction_id: str = field(default_factory=lambda: ''.join(random.choices(string.ascii_letters + string.digits, k=20)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def calculate_hash(self) -> str:
        """Calculate transaction hash excluding signature"""
        tx_copy = self.to_dict()
        tx_copy.pop('signature', None)
        tx_string = json.dumps(tx_copy, sort_keys=True).encode()
        return hashlib.sha256(tx_string).hexdigest()

@dataclass
class Block:
    index: int
    timestamp: float
    transactions: List[Dict[str, Any]]
    proof: int
    previous_hash: str
    merkle_root: Optional[str] = None
    nonce: int = 0
    difficulty: int = 4
    block_reward: float = 50.0
    version: str = "1.0"
    miner: str = "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        return cls(**data)


class Wallet:
    def __init__(self, name: str = None):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.name = name or f"User-{uuid.uuid4().hex[:8]}"

    def get_public_key_string(self) -> str:
        """Get public key as PEM string"""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return base64.b64encode(pem).decode('utf-8')

    def sign_transaction(self, transaction: Transaction) -> str:
        """Sign a transaction with the private key"""
        transaction_hash = transaction.calculate_hash().encode()
        signature = self.private_key.sign(
            transaction_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')


class Blockchain:
    def __init__(self, difficulty: int = 4):
        
        self.chain = []
        self.current_transactions = []
        self.nodes = set()
        self.mining_reward = 50.0  
        self.difficulty = difficulty 
        self.target_block_time = 10  
        self.adjust_difficulty_blocks = 10  
        self.wallets = {}  
        self.pending_transactions = []  
        self.miner_thread = None
        self.mining = False
        self.blockchain_lock = threading.RLock()  

        
        self.admin_wallet = Wallet(name="Admin")
        self.wallets[self.admin_wallet.get_public_key_string()] = self.admin_wallet

        
        self.create_genesis_block()

    def create_genesis_block(self) -> None:
        """Create the genesis block with special parameters"""
        genesis_tx = Transaction(
            sender="0",
            recipient=self.admin_wallet.get_public_key_string(),
            amount=100.0
        )

        genesis_block = Block(
            index=1,
            timestamp=time.time(),
            transactions=[genesis_tx.to_dict()],
            proof=100,
            previous_hash="1",
            merkle_root=self.calculate_merkle_root([genesis_tx.to_dict()]),
            difficulty=self.difficulty,
            miner="Genesis"
        )

        self.chain.append(genesis_block)
        logger.info(f"Genesis block created: {genesis_block.to_dict()}")

    def register_node(self, address: str) -> None:
        """Add a new node to the list of nodes"""
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)
        logger.info(f"Node registered: {parsed_url.netloc}")

    def create_wallet(self, name: Optional[str] = None) -> Wallet:
        """Create a new wallet and return it"""
        wallet = Wallet(name=name)
        self.wallets[wallet.get_public_key_string()] = wallet
        logger.info(f"New wallet created: {wallet.name}")
        return wallet

    def get_balance(self, address: str) -> float:
        """Calculate balance for a given wallet address"""
        balance = 0.0

        
        for block in self.chain:
            for tx in block.transactions:
                if tx['recipient'] == address:
                    balance += tx['amount']
                if tx['sender'] == address:
                    balance -= tx['amount']

        return balance

    def calculate_merkle_root(self, transactions: List[Dict[str, Any]]) -> str:
        """Calculate the Merkle root of transactions for a block"""
        if not transactions:
            return hashlib.sha256("empty".encode()).hexdigest()

        
        tx_hashes = [hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest() for tx in transactions]

        
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 != 0:
                tx_hashes.append(tx_hashes[-1])  

            next_level = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i+1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())

            tx_hashes = next_level

        return tx_hashes[0]

    def new_transaction(self, sender: str, recipient: str, amount: float,
                        signature: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates a new transaction

        Returns the transaction if valid
        """
        if sender != "0":  
            sender_balance = self.get_balance(sender)
            if sender_balance < amount:
                raise ValueError(f"Insufficient balance: {sender_balance} < {amount}")

        transaction = Transaction(
            sender=sender,
            recipient=recipient,
            amount=amount,
            signature=signature
        )

        
        if sender != "0" and signature:
            if not self.verify_transaction(transaction):
                raise ValueError("Invalid transaction signature")

        self.pending_transactions.append(transaction.to_dict())
        logger.info(f"New transaction added: {transaction.to_dict()}")
        return transaction.to_dict()

    def verify_transaction(self, transaction: Transaction) -> bool:
        """Verify the signature of a transaction"""
        if transaction.sender == "0":  
            return True

        try:
           
            sender_key_pem = base64.b64decode(transaction.sender)
            public_key = serialization.load_pem_public_key(sender_key_pem)

            
            signature = base64.b64decode(transaction.signature)
            transaction_hash = transaction.calculate_hash().encode()

            public_key.verify(
                signature,
                transaction_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except (InvalidSignature, Exception) as e:
            logger.error(f"Signature verification failed: {str(e)}")
            return False

    def adjust_difficulty(self) -> None:
        """Adjust mining difficulty based on recent block times"""
        if len(self.chain) % self.adjust_difficulty_blocks != 0 or len(self.chain) <= self.adjust_difficulty_blocks:
            return

        
        recent_blocks = self.chain[-self.adjust_difficulty_blocks:]
        time_taken = recent_blocks[-1].timestamp - recent_blocks[0].timestamp
        avg_time_per_block = time_taken / (self.adjust_difficulty_blocks - 1)

       
        if avg_time_per_block < self.target_block_time * 0.8:
            self.difficulty += 1
            logger.info(f"Difficulty increased to {self.difficulty} (blocks too fast)")
        elif avg_time_per_block > self.target_block_time * 1.2:
            self.difficulty = max(1, self.difficulty - 1)
            logger.info(f"Difficulty decreased to {self.difficulty} (blocks too slow)")

    def proof_of_work(self, block: Block) -> Tuple[int, str]:
        """
        Proof of Work algorithm:
        - Find a nonce value so that hash(block) contains leading zeros equal to difficulty

        Returns:
            Tuple of (nonce, hash)
        """
        block_dict = block.to_dict()
        nonce = 0
        valid_hash = ""

        while True:
            if not self.mining:
                raise InterruptedError("Mining stopped")

            block_dict['nonce'] = nonce
            block_string = json.dumps(block_dict, sort_keys=True).encode()
            new_hash = hashlib.sha256(block_string).hexdigest()

            if new_hash[:block.difficulty] == '0' * block.difficulty:
                valid_hash = new_hash
                break

            nonce += 1

        return nonce, valid_hash

    def mine_block(self, miner_address: str) -> Optional[Block]:
        """
        Mine a new block with pending transactions

        Args:
            miner_address: Address to receive mining reward

        Returns:
            The newly mined block or None if mining was interrupted
        """
        if not self.mining:
            return None

        with self.blockchain_lock:
           
            if not self.pending_transactions:
                
                reward_tx = self.new_transaction(
                    sender="0",
                    recipient=miner_address,
                    amount=self.calculate_mining_reward()
                )
                transactions = [reward_tx]
            else:
                
                reward_tx = self.new_transaction(
                    sender="0",
                    recipient=miner_address,
                    amount=self.calculate_mining_reward()
                )

               
                transactions = [reward_tx] + self.pending_transactions[:9]
                
                self.pending_transactions = self.pending_transactions[9:] if len(self.pending_transactions) > 9 else []

            
            last_block = self.chain[-1]
            merkle_root = self.calculate_merkle_root(transactions)

            new_block = Block(
                index=last_block.index + 1,
                timestamp=time.time(),
                transactions=transactions,
                proof=0,  
                previous_hash=self.hash(last_block),
                merkle_root=merkle_root,
                difficulty=self.difficulty,
                miner=miner_address
            )

            try:
                
                nonce, block_hash = self.proof_of_work(new_block)
                new_block.nonce = nonce
                new_block.proof = int(block_hash, 16) % 10**8  

               
                self.chain.append(new_block)
                logger.info(f"New block mined: {new_block.to_dict()}")

                
                self.adjust_difficulty()

                return new_block
            except InterruptedError:
                logger.info("Mining interrupted")
                return None

    def start_mining(self, miner_address: str) -> None:
        """Start mining in a separate thread"""
        if self.mining:
            return

        self.mining = True

        def mining_thread():
            logger.info(f"Mining started for miner: {miner_address}")
            while self.mining:
                try:
                    mined_block = self.mine_block(miner_address)
                    if mined_block:
                       
                        self.broadcast_new_block(mined_block)
                except Exception as e:
                    logger.error(f"Mining error: {str(e)}")
                    time.sleep(5)  
        self.miner_thread = threading.Thread(target=mining_thread)
        self.miner_thread.daemon = True
        self.miner_thread.start()

    def stop_mining(self) -> None:
        """Stop the mining thread"""
        self.mining = False
        if self.miner_thread:
            self.miner_thread.join(timeout=1)
            logger.info("Mining stopped")

    def calculate_mining_reward(self) -> float:
        """Calculate mining reward with halving"""
        halvings = len(self.chain) // 210000
        return self.mining_reward / (2 ** halvings)

    def broadcast_new_block(self, block: Block) -> None:
        """Broadcast a new block to all registered nodes"""
        for node in self.nodes:
            try:
                requests.post(f'http://{node}/blocks/receive',
                              json={'block': block.to_dict()})
            except Exception as e:
                logger.error(f"Failed to broadcast to node {node}: {str(e)}")

    def valid_chain(self, chain: List[Dict[str, Any]]) -> bool:
        """
        Determine if a given blockchain is valid

        Args:
            chain: List of blocks

        Returns:
            True if valid, False if not
        """
        blocks = [Block.from_dict(block) for block in chain]

        for i in range(1, len(blocks)):
            current_block = blocks[i]
            previous_block = blocks[i-1]

            if current_block.previous_hash != self.hash(previous_block):
                logger.error(f"Invalid previous hash at block {current_block.index}")
                return False

            block_hash = self.hash(current_block)
            if block_hash[:current_block.difficulty] != '0' * current_block.difficulty:
                logger.error(f"Invalid proof of work at block {current_block.index}")
                return False

            if current_block.merkle_root != self.calculate_merkle_root(current_block.transactions):
                logger.error(f"Invalid merkle root at block {current_block.index}")
                return False

        return True

    def resolve_conflicts(self) -> bool:
        """
        Consensus algorithm:
        - Replace our chain with the longest valid chain in the network

        Returns:
            True if our chain was replaced, False if not
        """
        replaced = False
        max_length = len(self.chain)

        for node in self.nodes:
            try:
                response = requests.get(f'http://{node}/chain')

                if response.status_code == 200:
                    data = response.json()
                    length = data['length']
                    chain = data['chain']

                    if length > max_length and self.valid_chain(chain):
                        max_length = length
                        with self.blockchain_lock:
                            self.chain = [Block.from_dict(block) for block in chain]
                            replaced = True
            except Exception as e:
                logger.error(f"Error fetching chain from {node}: {str(e)}")

        if replaced:
            logger.info("Chain replaced with longer chain from network")

        return replaced

    @staticmethod
    def hash(block: Block) -> str:
        """
        Create a SHA-256 hash of a Block

        Args:
            block: Block object

        Returns:
            Hash string
        """
        block_string = json.dumps(block.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def get_block_by_index(self, index: int) -> Optional[Block]:
        """Get a block by its index"""
        for block in self.chain:
            if block.index == index:
                return block
        return None

    def get_transaction_history(self, address: str) -> List[Dict[str, Any]]:
        """Get transaction history for a specific address"""
        history = []

        for block in self.chain:
            for tx in block.transactions:
                if tx['sender'] == address or tx['recipient'] == address:
                    tx_with_block = tx.copy()
                    tx_with_block['block_index'] = block.index
                    tx_with_block['block_timestamp'] = block.timestamp
                    history.append(tx_with_block)

        return sorted(history, key=lambda x: x['timestamp'], reverse=True)

    def generate_blockchain_visualization(self) -> str:
        """Generate a visualization of the blockchain as a PNG image in base64"""
        G = nx.DiGraph()

        for i, block in enumerate(self.chain):
            G.add_node(f"Block {block.index}", shape='box', label=f"Block {block.index}\nHash: {self.hash(block)[:6]}...")

            if i > 0:
                G.add_edge(f"Block {self.chain[i-1].index}", f"Block {block.index}")

        for block in self.chain[-3:]:
            for i, tx in enumerate(block.transactions):
                tx_id = f"Tx {tx.get('transaction_id', i)[:6]}"
                G.add_node(tx_id, shape='ellipse', label=f"{tx['sender'][:6]}... → {tx['recipient'][:6]}...\nAmount: {tx['amount']}")
                G.add_edge(f"Block {block.index}", tx_id)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000,
                font_size=8, font_weight='bold', arrows=True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return img_base64

    def to_dict(self) -> Dict[str, Any]:
        """Convert blockchain to dictionary for API responses"""
        return {
            'chain': [block.to_dict() for block in self.chain],
            'length': len(self.chain),
            'pending_transactions': len(self.pending_transactions),
            'nodes': list(self.nodes),
            'difficulty': self.difficulty,
            'mining_reward': self.calculate_mining_reward()
        }


app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)
socketio = SocketIO(app)

blockchain = Blockchain(difficulty=4)

@app.route('/')
def index():
    """Render the main dashboard"""
    stats = {
        'blocks': len(blockchain.chain),
        'transactions': sum(len(block.transactions) for block in blockchain.chain),
        'nodes': len(blockchain.nodes),
        'difficulty': blockchain.difficulty,
        'pending_tx': len(blockchain.pending_transactions),
        'latest_block_time': datetime.fromtimestamp(blockchain.chain[-1].timestamp).strftime('%Y-%m-%d %H:%M:%S'),
        'mining_status': "Active" if blockchain.mining else "Inactive"
    }

    blockchain_graph = blockchain.generate_blockchain_visualization()

    current_wallet = None
    wallet_address = session.get('wallet_address')
    if wallet_address and wallet_address in blockchain.wallets:
        current_wallet = {
            'address': wallet_address,
            'name': blockchain.wallets[wallet_address].name,
            'balance': blockchain.get_balance(wallet_address)
        }

    return render_template('index.html',
                           stats=stats,
                           blockchain_graph=blockchain_graph,
                           wallet=current_wallet,
                           recent_blocks=blockchain.chain[-5:])

@app.route('/wallets/create', methods=['POST'])
def create_wallet():
    """Create a new wallet"""
    name = request.form.get('name', '')
    wallet = blockchain.create_wallet(name)

    address = wallet.get_public_key_string()
    session['wallet_address'] = address

    flash(f'New wallet created: {name}', 'success')
    return redirect(url_for('index'))

@app.route('/wallets/select', methods=['POST'])
def select_wallet():
    """Select an existing wallet"""
    address = request.form.get('address')
    if address in blockchain.wallets:
        session['wallet_address'] = address
        flash(f'Wallet selected: {blockchain.wallets[address].name}', 'success')
    else:
        flash('Invalid wallet', 'error')

    return redirect(url_for('index'))

@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    """Create a new transaction"""
    sender = session.get('wallet_address')
    if not sender:
        flash('Please select or create a wallet first', 'error')
        return redirect(url_for('index'))

    recipient = request.form.get('recipient')
    amount = float(request.form.get('amount'))

    try:
        wallet = blockchain.wallets[sender]
        tx = Transaction(sender=sender, recipient=recipient, amount=amount)
        tx.signature = wallet.sign_transaction(tx)

        blockchain.new_transaction(tx.sender, tx.recipient, tx.amount, tx.signature)
        flash('Transaction created successfully', 'success')

        socketio.emit('new_transaction', tx.to_dict())
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')

    return redirect(url_for('index'))

@app.route('/mine/start', methods=['POST'])
def start_mining():
    """Start mining process"""
    address = session.get('wallet_address')
    if not address:
        flash('Please select or create a wallet first', 'error')
        return redirect(url_for('index'))

    blockchain.start_mining(address)
    flash('Mining started', 'success')
    return redirect(url_for('index'))

@app.route('/mine/stop', methods=['POST'])
def stop_mining():
    """Stop mining process"""
    blockchain.stop_mining()
    flash('Mining stopped', 'success')
    return redirect(url_for('index'))

@app.route('/chain', methods=['GET'])
def get_chain():
    """Return the full blockchain"""
    return jsonify(blockchain.to_dict()), 200

@app.route('/blocks/<int:index>', methods=['GET'])
def get_block(index):
    """Get a specific block by index"""
    block = blockchain.get_block_by_index(index)
    if block:
        return jsonify(block.to_dict()), 200
    return jsonify({'error': 'Block not found'}), 404

@app.route('/blocks/receive', methods=['POST'])
def receive_block():
    """Receive a new block from another node"""
    data = request.get_json()
    if not data or 'block' not in data:
        return jsonify({'error': 'Invalid block data'}), 400

    block_data = data['block']
    block = Block.from_dict(block_data)

    if blockchain.valid_chain([b.to_dict() for b in blockchain.chain] + [block.to_dict()]):
        blockchain.chain.append(block)
        socketio.emit('new_block', block.to_dict())
        return jsonify({'message': 'Block accepted'}), 201

    return jsonify({'error': 'Invalid block'}), 400

@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    """Register new nodes"""
    data = request.get_json()
    nodes = data.get('nodes')

    if not nodes:
        return jsonify({'error': 'Please provide a list of nodes'}), 400

    for node in nodes:
        blockchain.register_node(node)

    return jsonify({
        'message': 'Nodes registered successfully',
        'total_nodes': list(blockchain.nodes)
    }), 201

@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    """Resolve conflicts using consensus algorithm"""
    replaced = blockchain.resolve_conflicts()

    if replaced:
        response = {
            'message': 'Chain replaced with longer chain',
            'new_chain': [block.to_dict() for block in blockchain.chain]
        }
    else:
        response = {
            'message': 'Current chain is authoritative',
            'chain': [block.to_dict() for block in blockchain.chain]
        }

    return jsonify(response), 200

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get blockchain statistics"""
    stats = {
        'blocks': len(blockchain.chain),
        'transactions': sum(len(block.transactions) for block in blockchain.chain),
        'nodes': len(blockchain.nodes),
        'difficulty': blockchain.difficulty,
        'pending_tx': len(blockchain.pending_transactions),
        'last_block_time': blockchain.chain[-1].timestamp,
        'mining_reward': blockchain.calculate_mining_reward()
    }
    return jsonify(stats), 200

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    emit('blockchain_update', blockchain.to_dict())

@app.before_request
def before_request():
    """Create templates directory and files if they don't exist"""
    if not os.path.exists('templates'):
        os.makedirs('templates')

    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Blockchain Explorer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .navbar {
            background: linear-gradient(135deg, #3a6186, #89253e);
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s;
            margin-bottom: 20px;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card-header {
            background: linear-gradient(135deg, #3a6186, #89253e);
            color: white;
            border-radius: 10px 10px 0 0 !important;
        }
        .stats-icon {
            font-size: 2.5rem;
            color: #3a6186;
        }
        .blockchain-visualization {
            width: 100%;
            max-height: 500px;
            object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .block-card {
            background: linear-gradient(135deg, #a8edea, #fed6e3);
            border: none;
        }
        .transaction-badge {
            background-color: #3a6186;
            color: white;
        }
        .btn-primary {
        background: linear-gradient(135deg, #3a6186, #fed6e3);
        }
        </style>
        <!-- more HTML content -->
        """)

