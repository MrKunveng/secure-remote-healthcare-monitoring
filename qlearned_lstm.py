#!/usr/bin/env python3
"""
Q-Learning + LSTM Integration for Remote Healthcare Monitoring
=============================================================

This implementation combines Q-learning with LSTM neural networks to improve
decision making for medical monitoring. The LSTM learns temporal patterns in
patient vitals while Q-learning optimizes the action selection policy.

Actions:
- 0: Monitor (continue observation)
- 1: Administer Medication (medical intervention)

Architecture:
- LSTM: Learns temporal patterns in patient vitals
- Q-Learning: Optimizes action selection based on LSTM features
- Hybrid: Combines both approaches for enhanced performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
import os
from datetime import datetime, timedelta
import joblib

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    print(f"PyTorch version: {torch.__version__}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
except ImportError:
    print("PyTorch not found. Please install it using: pip install torch")
    raise

warnings.filterwarnings('ignore')


class MedicalLSTM(nn.Module):
    """PyTorch LSTM model for medical decision making"""
    
    def __init__(self, input_size, sequence_length, hidden_size=64, num_layers=2, dropout=0.2):
        super(MedicalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout3 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 1)
        
        # Move to device
        self.to(device)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # First LSTM layer
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # Second LSTM layer (only take last output)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Take only the last timestep output
        last_output = lstm_out2[:, -1, :]
        
        # Dense layers
        x = F.relu(self.fc1(last_output))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x


class EarlyStopping:
    """Early stopping utility class"""
    def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def load_lstm_model(model_path: str, info_path: str) -> MedicalLSTM:
    """Load a saved PyTorch LSTM model"""
    # Load model info
    model_info = torch.load(info_path, map_location=device)
    
    # Create model with saved parameters
    model = MedicalLSTM(
        input_size=model_info['input_size'],
        sequence_length=model_info['sequence_length'],
        hidden_size=model_info['hidden_size'],
        num_layers=model_info['num_layers'],
        dropout=model_info['dropout']
    )
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


class MedicalEnvironmentLSTM:
    """Enhanced environment with LSTM integration"""
    
    def __init__(self, data_path: str, n_states: int = 100, sequence_length: int = 10):
        self.data_path = data_path
        self.n_states = n_states
        self.n_actions = 2
        self.sequence_length = sequence_length
        
        # Load and preprocess data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape: {self.df.shape}")
        
        self._preprocess_data()
        self._create_sequences()
        self._create_state_space()
        self._build_lstm_model()
        
        # Environment state
        self.current_episode = 0
        self.current_state = 0
        self.done = False
        
    def _preprocess_data(self):
        """Preprocess the medical dataset"""
        # Select relevant features
        potential_features = [
            'Age', 'Temperature_C', 'Heart Rate_bpm', 'Respiration Rate_brpm',
            'SpO2_percent', 'BMI', 'Sleep Duration_hours', 'Stress Level',
            'Blood Oxygen Level (%)'
        ]
        
        self.feature_columns = [col for col in potential_features if col in self.df.columns]
        
        # Handle blood pressure
        if 'Blood Pressure_mmHg' in self.df.columns:
            bp_split = self.df['Blood Pressure_mmHg'].str.split('/', expand=True)
            self.df['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
            self.df['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
            self.feature_columns.extend(['Systolic_BP', 'Diastolic_BP'])
        
        # Encode categorical variables
        if 'Gender' in self.df.columns:
            le_gender = LabelEncoder()
            self.df['Gender_encoded'] = le_gender.fit_transform(self.df['Gender'].fillna('Unknown'))
            self.feature_columns.append('Gender_encoded')
            
        if 'Activity Level' in self.df.columns:
            le_activity = LabelEncoder()
            self.df['Activity_encoded'] = le_activity.fit_transform(self.df['Activity Level'].fillna('Unknown'))
            self.feature_columns.append('Activity_encoded')
        
        # Encode target action
        if 'Action' in self.df.columns:
            self.df['Action_encoded'] = (self.df['Action'] == 'Administer Medication').astype(int)
        else:
            raise ValueError("No 'Action' column found in dataset")
        
        # Handle missing values
        for col in self.feature_columns:
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(0)
        
        # Sort by patient and date for temporal sequences
        if 'Patient ID' in self.df.columns and 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df = self.df.sort_values(['Patient ID', 'Date'])
        
        # Normalize features for LSTM
        self.scaler = MinMaxScaler()
        self.features_normalized = self.scaler.fit_transform(self.df[self.feature_columns])
        
        print(f"Using features: {self.feature_columns}")
        print(f"Dataset shape after preprocessing: {self.df.shape}")
        
    def _create_sequences(self):
        """Create temporal sequences for LSTM"""
        sequences = []
        targets = []
        patient_sequences = []
        
        # Group by patient to create sequences
        if 'Patient ID' in self.df.columns:
            for patient_id in self.df['Patient ID'].unique():
                patient_data = self.df[self.df['Patient ID'] == patient_id]
                patient_features = self.features_normalized[self.df['Patient ID'] == patient_id]
                patient_actions = patient_data['Action_encoded'].values
                
                # Create sequences for this patient
                for i in range(len(patient_features) - self.sequence_length + 1):
                    seq = patient_features[i:i + self.sequence_length]
                    target = patient_actions[i + self.sequence_length - 1]
                    sequences.append(seq)
                    targets.append(target)
                    patient_sequences.append(patient_id)
        else:
            # If no patient ID, create sequences from the entire dataset
            for i in range(len(self.features_normalized) - self.sequence_length + 1):
                seq = self.features_normalized[i:i + self.sequence_length]
                target = self.df['Action_encoded'].iloc[i + self.sequence_length - 1]
                sequences.append(seq)
                targets.append(target)
        
        self.sequences = np.array(sequences)
        self.sequence_targets = np.array(targets)
        self.patient_sequences = patient_sequences if patient_sequences else None
        
        print(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        print(f"Sequence shape: {self.sequences.shape}")
        
    def _create_state_space(self):
        """Create discrete state space using K-means clustering on sequence features"""
        # Use the last timestep of each sequence for state representation
        last_timestep_features = self.sequences[:, -1, :]
        
        # Ensure n_states doesn't exceed number of samples
        n_samples = len(last_timestep_features)
        self.n_states = min(self.n_states, n_samples)
        
        self.kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
        self.states = self.kmeans.fit_predict(last_timestep_features)
        
        # Create observation and action space properties
        self.observation_space = type('obj', (object,), {'n': self.n_states})()
        self.action_space = type('obj', (object,), {
            'n': self.n_actions,
            'sample': lambda: np.random.randint(0, self.n_actions)
        })()
        
        print(f"Created {self.n_states} states from sequence features")
        
    def _build_lstm_model(self):
        """Build LSTM model for temporal pattern learning"""
        self.lstm_model = MedicalLSTM(
            input_size=len(self.feature_columns),
            sequence_length=self.sequence_length,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
        )
        
        print("LSTM model built successfully")
        print(f"Model architecture:\n{self.lstm_model}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.lstm_model.parameters())
        trainable_params = sum(p.numel() for p in self.lstm_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def train_lstm(self, epochs: int = 1000, validation_split: float = 0.2, batch_size: int = 32):
        """Train the LSTM model"""
        print("Training LSTM model...")
        
        # Convert data to tensors
        X_tensor = torch.FloatTensor(self.sequences).to(device)
        y_tensor = torch.FloatTensor(self.sequence_targets).unsqueeze(1).to(device)
        
        # Split into train and validation
        n_train = int(len(X_tensor) * (1 - validation_split))
        X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
        y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
        
        # Training history
        history = {
            'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [],
            'precision': [], 'val_precision': [], 'recall': [], 'val_recall': []
        }
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        for epoch in range(epochs):
            # Training phase
            self.lstm_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_tp = train_fp = train_fn = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate metrics
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                # Precision/Recall metrics
                train_tp += ((predicted == 1) & (batch_y == 1)).sum().item()
                train_fp += ((predicted == 1) & (batch_y == 0)).sum().item()
                train_fn += ((predicted == 0) & (batch_y == 1)).sum().item()
            
            # Validation phase
            self.lstm_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_tp = val_fp = val_fn = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.lstm_model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    # Calculate metrics
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    # Precision/Recall metrics
                    val_tp += ((predicted == 1) & (batch_y == 1)).sum().item()
                    val_fp += ((predicted == 1) & (batch_y == 0)).sum().item()
                    val_fn += ((predicted == 0) & (batch_y == 1)).sum().item()
            
            # Calculate epoch metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
            train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
            val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
            val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
            
            # Update history
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['precision'].append(train_precision)
            history['val_precision'].append(val_precision)
            history['recall'].append(train_recall)
            history['val_recall'].append(val_recall)
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}")
            
            # Early stopping
            if early_stopping(val_loss, self.lstm_model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.lstm_history = type('History', (), {'history': history})()
        
        print("LSTM training completed")
        print(f"Final - Train Loss: {history['loss'][-1]:.4f}, Train Acc: {history['accuracy'][-1]:.4f}")
        print(f"Final - Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {history['val_accuracy'][-1]:.4f}")
        
        return self.lstm_history
    
    def get_lstm_features(self, sequence_idx: int) -> np.ndarray:
        """Get LSTM-derived features for a sequence"""
        if sequence_idx >= len(self.sequences):
            sequence_idx = np.random.randint(0, len(self.sequences))
        
        # Convert to tensor and move to device
        sequence = torch.FloatTensor(self.sequences[sequence_idx:sequence_idx+1]).to(device)
        
        # Get prediction in evaluation mode
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_output = self.lstm_model(sequence)
        
        # Convert back to numpy and return
        return lstm_output.cpu().numpy().flatten()
    
    def reset(self) -> int:
        """Reset environment and return initial state"""
        self.current_episode = np.random.randint(0, len(self.sequences))
        self.current_state = self.states[self.current_episode]
        self.done = False
        return int(self.current_state)
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Execute action and return next state, reward, done, info"""
        true_action = self.sequence_targets[self.current_episode]
        reward = self._calculate_reward(action, true_action)
        
        self.done = True
        next_state = self.current_state
        
        # Get LSTM prediction for additional info
        lstm_features = self.get_lstm_features(self.current_episode)
        lstm_prediction = int(lstm_features[0] > 0.5)
        
        info = {
            'true_action': true_action,
            'predicted_action': action,
            'lstm_prediction': lstm_prediction,
            'lstm_confidence': float(lstm_features[0]),
            'sequence_idx': self.current_episode
        }
        
        return int(next_state), reward, self.done, info
    
    def _calculate_reward(self, predicted_action: int, true_action: int) -> float:
        """Calculate reward based on prediction accuracy and medical priorities"""
        if predicted_action == true_action:
            if true_action == 0:  # Correct Monitor
                return 5.0
            else:  # Correct Medicate
                return 10.0
        else:
            if predicted_action == 1 and true_action == 0:  # False Positive
                return -20.0
            else:  # False Negative
                return -5.0


class HybridQLSTMAgent:
    """Hybrid agent combining Q-learning with LSTM features"""
    
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, 
                 gamma: float = 0.6, epsilon: float = 0.1, use_lstm: bool = True):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_lstm = use_lstm
        
        # Initialize Q-table
        self.Q = np.zeros([n_states, n_actions])
        
        # LSTM integration weights
        self.lstm_weight = 0.3  # How much to weight LSTM vs Q-learning
        
    def choose_action(self, state: int, lstm_confidence: float = None, 
                     lstm_prediction: int = None) -> int:
        """Choose action using hybrid approach"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)  # Explore
        
        # Q-learning action
        q_action = np.argmax(self.Q[state])
        
        if not self.use_lstm or lstm_confidence is None:
            return q_action
        
        # Hybrid decision: combine Q-learning with LSTM
        q_values = self.Q[state].copy()
        
        # Boost the action suggested by LSTM based on confidence
        if lstm_prediction is not None:
            lstm_boost = self.lstm_weight * lstm_confidence
            q_values[lstm_prediction] += lstm_boost
        
        return np.argmax(q_values)
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-value using Q-learning update rule"""
        old_value = self.Q[state, action]
        next_max = np.max(self.Q[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.Q[state, action] = new_value
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """Decay exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


def train_hybrid_agent(env: MedicalEnvironmentLSTM, agent: HybridQLSTMAgent, 
                      episodes: int = 1000) -> dict:
    """Train hybrid Q-learning + LSTM agent"""
    
    # Metrics tracking
    episode_rewards = []
    episode_accuracies = []
    cumulative_rewards = []
    success_rate_history = []
    lstm_agreement = []  # Track when Q-learning and LSTM agree
    
    # Confusion matrix tracking
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    print("Starting Hybrid Q-Learning + LSTM Training...")
    
    for episode in range(1, episodes + 1):
        try:
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                # Get LSTM features for current sequence
                lstm_features = env.get_lstm_features(env.current_episode)
                lstm_confidence = float(lstm_features[0])
                lstm_prediction = int(lstm_confidence > 0.5)
                
                # Choose action using hybrid approach
                action = agent.choose_action(state, lstm_confidence, lstm_prediction)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Update Q-value
                agent.update_q_value(state, action, reward, next_state)
                
                # Update metrics
                total_reward += reward
                
                # Track LSTM agreement
                q_only_action = np.argmax(agent.Q[state])
                lstm_agreement.append(int(q_only_action == lstm_prediction))
                
                # Update confusion matrix
                true_action = info['true_action']
                if action == 1 and true_action == 1:
                    true_positives += 1
                elif action == 0 and true_action == 0:
                    true_negatives += 1
                elif action == 1 and true_action == 0:
                    false_positives += 1
                elif action == 0 and true_action == 1:
                    false_negatives += 1
                
                state = next_state
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Track metrics
            episode_rewards.append(total_reward)
            accuracy = (true_positives + true_negatives) / max(1, episode)
            episode_accuracies.append(accuracy)
            cumulative_rewards.append(sum(episode_rewards))
            
            # Calculate success rate
            total_predictions = true_positives + true_negatives + false_positives + false_negatives
            success_rate = (true_positives + true_negatives) / max(1, total_predictions)
            success_rate_history.append(success_rate)
            
            # Print progress
            if episode % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_agreement = np.mean(lstm_agreement[-100:]) if len(lstm_agreement) >= 100 else np.mean(lstm_agreement)
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Accuracy = {accuracy:.3f}, Success Rate = {success_rate:.3f}, "
                      f"LSTM Agreement = {avg_agreement:.3f}")
                      
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            continue
    
    # Final metrics
    final_accuracy = (true_positives + true_negatives) / episodes if episodes > 0 else 0
    final_success_rate = success_rate_history[-1] if success_rate_history else 0
    
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_accuracies': episode_accuracies,
        'cumulative_rewards': cumulative_rewards,
        'success_rate_history': success_rate_history,
        'lstm_agreement': lstm_agreement,
        'final_accuracy': final_accuracy,
        'final_success_rate': final_success_rate,
        'confusion_matrix': {
            'TP': true_positives,
            'TN': true_negatives,
            'FP': false_positives,
            'FN': false_negatives
        }
    }
    
    return metrics


def evaluate_hybrid_agent(env: MedicalEnvironmentLSTM, agent: HybridQLSTMAgent, 
                         test_episodes: int = 1000) -> dict:
    """Evaluate hybrid agent"""
    
    test_rewards = []
    predictions = []
    true_labels = []
    lstm_predictions = []
    q_only_predictions = []
    
    # Temporarily set epsilon to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(test_episodes):
        try:
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                # Get LSTM features
                lstm_features = env.get_lstm_features(env.current_episode)
                lstm_confidence = float(lstm_features[0])
                lstm_prediction = int(lstm_confidence > 0.5)
                
                # Get Q-only action for comparison
                q_only_action = np.argmax(agent.Q[state])
                
                # Get hybrid action
                action = agent.choose_action(state, lstm_confidence, lstm_prediction)
                
                next_state, reward, done, info = env.step(action)
                
                total_reward += reward
                predictions.append(action)
                true_labels.append(info['true_action'])
                lstm_predictions.append(lstm_prediction)
                q_only_predictions.append(q_only_action)
                
                state = next_state
            
            test_rewards.append(total_reward)
            
        except Exception as e:
            print(f"Error in evaluation episode {episode}: {e}")
            continue
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    if not predictions:
        return {
            'accuracy': 0, 'average_reward': 0, 'precision': 0, 'recall': 0, 
            'specificity': 0, 'lstm_accuracy': 0, 'q_only_accuracy': 0,
            'confusion_matrix': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        }
    
    # Calculate evaluation metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    lstm_predictions = np.array(lstm_predictions)
    q_only_predictions = np.array(q_only_predictions)
    
    # Hybrid model metrics
    accuracy = np.mean(predictions == true_labels)
    avg_reward = np.mean(test_rewards)
    
    # Individual model accuracies
    lstm_accuracy = np.mean(lstm_predictions == true_labels)
    q_only_accuracy = np.mean(q_only_predictions == true_labels)
    
    # Confusion matrix for hybrid model
    tn = np.sum((predictions == 0) & (true_labels == 0))
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    
    # Calculate rates
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'average_reward': avg_reward,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'lstm_accuracy': lstm_accuracy,
        'q_only_accuracy': q_only_accuracy,
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    }


def compare_models(q_only_metrics: dict, hybrid_metrics: dict) -> dict:
    """Compare Q-learning only vs Q-learning + LSTM"""
    
    comparison = {
        'cumulative_reward': {
            'q_only': q_only_metrics['cumulative_rewards'][-1] if q_only_metrics['cumulative_rewards'] else 0,
            'hybrid': hybrid_metrics['cumulative_rewards'][-1] if hybrid_metrics['cumulative_rewards'] else 0
        },
        'success_rate': {
            'q_only': q_only_metrics['final_success_rate'],
            'hybrid': hybrid_metrics['final_success_rate']
        },
        'average_reward': {
            'q_only': np.mean(q_only_metrics['episode_rewards']) if q_only_metrics['episode_rewards'] else 0,
            'hybrid': np.mean(hybrid_metrics['episode_rewards']) if hybrid_metrics['episode_rewards'] else 0
        },
        'sample_efficiency': {
            'q_only': q_only_metrics['final_accuracy'],
            'hybrid': hybrid_metrics['final_accuracy']
        }
    }
    
    # Calculate improvements
    for metric in comparison:
        q_val = comparison[metric]['q_only']
        h_val = comparison[metric]['hybrid']
        improvement = ((h_val - q_val) / max(abs(q_val), 1e-8)) * 100
        comparison[metric]['improvement_pct'] = improvement
    
    return comparison


def plot_comparison_metrics(comparison: dict, save_path: str = None):
    """Plot comparison between Q-only and Hybrid models"""
    
    metrics = list(comparison.keys())
    q_only_values = [comparison[m]['q_only'] for m in metrics]
    hybrid_values = [comparison[m]['hybrid'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart comparison
    ax1.bar(x - width/2, q_only_values, width, label='Q-Learning Only', alpha=0.8)
    ax1.bar(x + width/2, hybrid_values, width, label='Q-Learning + LSTM', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement percentage
    improvements = [comparison[m]['improvement_pct'] for m in metrics]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    ax2.bar(range(len(metrics)), improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Hybrid Model Improvement over Q-Learning Only')
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def plot_lstm_predictions(env: MedicalEnvironmentLSTM, save_path: str = None):
    """Plot LSTM actual vs predicted values"""
    
    # Get predictions for all sequences
    env.lstm_model.eval()
    with torch.no_grad():
        sequences_tensor = torch.FloatTensor(env.sequences).to(device)
        lstm_predictions = env.lstm_model(sequences_tensor).cpu().numpy()
    actual_values = env.sequence_targets
    
    # Create time indices (assuming sequential data)
    time_indices = range(len(actual_values))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_indices, actual_values, label='Actual Actions', alpha=0.7, linewidth=2)
    plt.plot(time_indices, lstm_predictions.flatten(), label='LSTM Predictions', alpha=0.7, linewidth=2)
    plt.title('LSTM: Actual vs Predicted Medical Actions')
    plt.xlabel('Sequence Index')
    plt.ylabel('Action Probability (0=Monitor, 1=Medicate)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"LSTM predictions plot saved to: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path: str = None):
    """Plot LSTM training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to run Q-learning + LSTM integration"""
    
    try:
        # Configuration
        data_path = "/Users/lukman/Desktop/Research/medical_dataset.csv"
        n_states = 100
        n_episodes = 1000
        sequence_length = 10
        
        print("=== Q-Learning + LSTM Integration for Medical Monitoring ===")
        print("Initializing Enhanced Medical Environment with LSTM...")
        
        # Create enhanced environment
        env = MedicalEnvironmentLSTM(data_path, n_states=n_states, sequence_length=sequence_length)
        
        # Train LSTM first
        print("\n1. Training LSTM model...")
        lstm_history = env.train_lstm(epochs=1000, batch_size=32)
        
        # Plot LSTM training history
        print("\n2. Visualizing LSTM training history...")
        plot_training_history(lstm_history, save_path="lstm_training_history.png")
        
        # Plot LSTM predictions
        print("\n3. Visualizing LSTM predictions...")
        plot_lstm_predictions(env, save_path="lstm_predictions.png")
        
        # Create Q-learning only agent (for comparison)
        print("\n4. Training Q-Learning only agent...")
        q_only_agent = HybridQLSTMAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            alpha=0.1, gamma=0.9, epsilon=0.3,
            use_lstm=False  # Q-learning only
        )
        
        q_only_metrics = train_hybrid_agent(env, q_only_agent, episodes=n_episodes)
        
        # Create hybrid agent
        print("\n5. Training Hybrid Q-Learning + LSTM agent...")
        hybrid_agent = HybridQLSTMAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            alpha=0.1, gamma=0.9, epsilon=0.3,
            use_lstm=True  # Hybrid approach
        )
        
        hybrid_metrics = train_hybrid_agent(env, hybrid_agent, episodes=n_episodes)
        
        # Evaluate both models
        print("\n6. Evaluating models...")
        q_only_eval = evaluate_hybrid_agent(env, q_only_agent, test_episodes=1000)
        hybrid_eval = evaluate_hybrid_agent(env, hybrid_agent, test_episodes=1000)
        
        # Print results
        print("\n=== RESULTS ===")
        print("\nQ-Learning Only:")
        print(f"  Final Accuracy: {q_only_metrics['final_accuracy']:.3f}")
        print(f"  Final Success Rate: {q_only_metrics['final_success_rate']:.3f}")
        print(f"  Cumulative Reward: {q_only_metrics['cumulative_rewards'][-1] if q_only_metrics['cumulative_rewards'] else 0:.2f}")
        print(f"  Average Reward: {np.mean(q_only_metrics['episode_rewards']) if q_only_metrics['episode_rewards'] else 0:.2f}")
        
        print("\nQ-Learning + LSTM Hybrid:")
        print(f"  Final Accuracy: {hybrid_metrics['final_accuracy']:.3f}")
        print(f"  Final Success Rate: {hybrid_metrics['final_success_rate']:.3f}")
        print(f"  Cumulative Reward: {hybrid_metrics['cumulative_rewards'][-1] if hybrid_metrics['cumulative_rewards'] else 0:.2f}")
        print(f"  Average Reward: {np.mean(hybrid_metrics['episode_rewards']) if hybrid_metrics['episode_rewards'] else 0:.2f}")
        
        print("\nEvaluation Results:")
        print(f"Q-Only Test Accuracy: {q_only_eval['accuracy']:.3f}")
        print(f"Hybrid Test Accuracy: {hybrid_eval['accuracy']:.3f}")
        print(f"LSTM Only Accuracy: {hybrid_eval['lstm_accuracy']:.3f}")
        
        # Compare models
        print("\n7. Comparing models...")
        comparison = compare_models(q_only_metrics, hybrid_metrics)
        
        print("\n=== MODEL COMPARISON ===")
        for metric, values in comparison.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Q-Learning Only: {values['q_only']:.3f}")
            print(f"  Q-Learning + LSTM: {values['hybrid']:.3f}")
            print(f"  Improvement: {values['improvement_pct']:.1f}%")
        
        # Generate plots
        print("\n8. Generating comparison plots...")
        plot_comparison_metrics(comparison, save_path="model_comparison.png")
        
        # Save results
        print("\n9. Saving results...")
        
        # Save metrics
        results_df = pd.DataFrame({
            'Metric': ['Cumulative Reward', 'Success Rate', 'Average Reward', 'Sample Efficiency'],
            'Q_Learning_Only': [comparison[m]['q_only'] for m in comparison.keys()],
            'Q_Learning_LSTM': [comparison[m]['hybrid'] for m in comparison.keys()],
            'Improvement_Percent': [comparison[m]['improvement_pct'] for m in comparison.keys()]
        })
        results_df.to_csv("model_comparison_results.csv", index=False)
        
        # Save models
        joblib.dump(q_only_agent.Q, "q_table_only.pkl")
        joblib.dump(hybrid_agent.Q, "q_table_hybrid.pkl")
        torch.save(env.lstm_model.state_dict(), "lstm_model.pth")
        
        # Also save model architecture info for loading later
        model_info = {
            'input_size': len(env.feature_columns),
            'sequence_length': env.sequence_length,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        }
        torch.save(model_info, "lstm_model_info.pth")
        
        print("\nResults saved:")
        print("- Model comparison: model_comparison_results.csv")
        print("- Q-tables: q_table_only.pkl, q_table_hybrid.pkl")
        print("- LSTM model: lstm_model.pth, lstm_model_info.pth")
        print("- Plots: lstm_training_history.png, lstm_predictions.png, model_comparison.png")
        
        return env, q_only_agent, hybrid_agent, comparison
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    env, q_only_agent, hybrid_agent, comparison = main()