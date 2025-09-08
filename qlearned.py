#!/usr/bin/env python3
"""
Q-Learning Algorithm for Remote Healthcare Monitoring
====================================================

This implementation uses Q-learning to determine whether to "Monitor" or 
"Administer Medication" based on patient vitals. The algorithm learns to 
avoid unnecessary interventions while ensuring proper medical care.

Actions:
- 0: Monitor (continue observation)
- 1: Administer Medication (medical intervention)

Reward System:
- Correct prediction: +10
- False positive (unnecessary medication): -20 (heavily penalized)
- False negative (missed medication need): -5
- Monitor when should monitor: +5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import seaborn as sns
from typing import Tuple, List
import warnings
import os
warnings.filterwarnings('ignore')

class MedicalEnvironment:
    """Custom environment for medical vitals monitoring"""
    
    def __init__(self, data_path: str, n_states: int = 100):
        self.data_path = data_path
        self.n_states = n_states
        self.n_actions = 2  # Monitor (0) or Medicate (1)
        
        # Load and preprocess data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        self._preprocess_data()
        self._create_state_space()
        
        # Environment state
        self.current_episode = 0
        self.current_state = 0
        self.done = False
        
    def _preprocess_data(self):
        """Preprocess the medical dataset"""
        # Check available columns
        available_columns = list(self.df.columns)
        print(f"Available columns: {available_columns}")
        
        # Select relevant features for state representation (only use available columns)
        potential_features = [
            'Age', 'Temperature_C', 'Heart Rate_bpm', 'Respiration Rate_brpm',
            'SpO2_percent', 'BMI', 'Sleep Duration_hours', 'Stress Level',
            'Blood Oxygen Level (%)'
        ]
        
        self.feature_columns = [col for col in potential_features if col in self.df.columns]
        print(f"Using features: {self.feature_columns}")
        
        # Handle blood pressure (extract systolic and diastolic)
        if 'Blood Pressure_mmHg' in self.df.columns:
            try:
                bp_split = self.df['Blood Pressure_mmHg'].str.split('/', expand=True)
                self.df['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
                self.df['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
                self.feature_columns.extend(['Systolic_BP', 'Diastolic_BP'])
                print("Blood pressure split into systolic and diastolic")
            except Exception as e:
                print(f"Warning: Could not process blood pressure: {e}")
        
        # Encode categorical variables
        if 'Gender' in self.df.columns:
            try:
                le_gender = LabelEncoder()
                self.df['Gender_encoded'] = le_gender.fit_transform(self.df['Gender'].fillna('Unknown'))
                self.feature_columns.append('Gender_encoded')
                print("Gender encoded successfully")
            except Exception as e:
                print(f"Warning: Could not encode gender: {e}")
            
        if 'Activity Level' in self.df.columns:
            try:
                le_activity = LabelEncoder()
                self.df['Activity_encoded'] = le_activity.fit_transform(self.df['Activity Level'].fillna('Unknown'))
                self.feature_columns.append('Activity_encoded')
                print("Activity level encoded successfully")
            except Exception as e:
                print(f"Warning: Could not encode activity level: {e}")
        
        # Encode target action
        if 'Action' in self.df.columns:
            self.df['Action_encoded'] = (self.df['Action'] == 'Administer Medication').astype(int)
            print(f"Action distribution: {self.df['Action_encoded'].value_counts()}")
        else:
            raise ValueError("No 'Action' column found in dataset")
        
        # Check for missing values and handle them
        print(f"Missing values before cleaning:")
        print(self.df[self.feature_columns + ['Action_encoded']].isnull().sum())
        
        # Fill missing values with median for numeric columns
        for col in self.feature_columns:
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 0)
        
        # Remove rows with missing target values
        self.df = self.df.dropna(subset=['Action_encoded'])
        
        print(f"Dataset shape after preprocessing: {self.df.shape}")
        
        # Ensure we have enough features
        if len(self.feature_columns) == 0:
            raise ValueError("No valid features found for state representation")
        
        # Normalize features
        try:
            self.scaler = StandardScaler()
            feature_data = self.df[self.feature_columns].values
            self.features_normalized = self.scaler.fit_transform(feature_data)
            print(f"Features normalized successfully. Shape: {self.features_normalized.shape}")
        except Exception as e:
            print(f"Error normalizing features: {e}")
            raise
        
    def _create_state_space(self):
        """Create discrete state space using K-means clustering"""
        try:
            # Ensure n_states doesn't exceed number of samples
            n_samples = len(self.features_normalized)
            self.n_states = min(self.n_states, n_samples)
            print(f"Using {self.n_states} states for {n_samples} samples")
            
            self.kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            self.states = self.kmeans.fit_predict(self.features_normalized)
            self.df['state'] = self.states
            
            print(f"State distribution: {np.bincount(self.states)}")
            
            # Create observation and action space properties
            self.observation_space = type('obj', (object,), {'n': self.n_states})()
            self.action_space = type('obj', (object,), {
                'n': self.n_actions,
                'sample': lambda: np.random.randint(0, self.n_actions)
            })()
            
        except Exception as e:
            print(f"Error creating state space: {e}")
            raise
        
    def reset(self) -> int:
        """Reset environment and return initial state"""
        self.current_episode = np.random.randint(0, len(self.df))
        self.current_state = self.df.iloc[self.current_episode]['state']
        self.done = False
        return int(self.current_state)
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Execute action and return next state, reward, done, info"""
        # Get true action from dataset
        true_action = self.df.iloc[self.current_episode]['Action_encoded']
        
        # Calculate reward based on action correctness
        reward = self._calculate_reward(action, true_action)
        
        # Episode ends after one step (single decision per patient state)
        self.done = True
        
        # Get next state (for Q-learning update, though episode ends)
        next_state = self.current_state  # Same state since episode ends
        
        info = {
            'true_action': true_action,
            'predicted_action': action,
            'patient_id': self.df.iloc[self.current_episode].get('Patient ID', 'Unknown')
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
            if predicted_action == 1 and true_action == 0:  # False Positive (unnecessary medication)
                return -20.0  # Heavy penalty for unnecessary intervention
            else:  # False Negative (missed medication need)
                return -5.0
    
    def render(self):
        """Render current state (optional visualization)"""
        if self.current_episode < len(self.df):
            patient_data = self.df.iloc[self.current_episode]
            print(f"Patient: {patient_data.get('Patient ID', 'Unknown')}")
            print(f"State: {self.current_state}")
            print(f"Vitals: HR={patient_data.get('Heart Rate_bpm', 'N/A')}, "
                  f"Temp={patient_data.get('Temperature_C', 'N/A')}, "
                  f"SpO2={patient_data.get('SpO2_percent', 'N/A')}")


class QLearningAgent:
    """Q-Learning Agent for medical decision making"""
    
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, 
                 gamma: float = 0.6, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table
        self.Q = np.zeros([n_states, n_actions])
        
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)  # Explore
        else:
            return np.argmax(self.Q[state])  # Exploit
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-value using Q-learning update rule"""
        old_value = self.Q[state, action]
        next_max = np.max(self.Q[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.Q[state, action] = new_value
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """Decay exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


def calculate_stepwise_returns(rewards: List[float], discount_factor: float = 0.99) -> np.ndarray:
    """Calculate discounted returns (numpy version, no torch dependency)"""
    if not rewards:
        return np.array([])
    
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = np.array(returns, dtype=np.float32)
    
    if len(returns) > 1 and returns.std() > 1e-8:
        normalized_returns = (returns - returns.mean()) / returns.std()
    else:
        normalized_returns = returns
    return normalized_returns


def calculate_loss(stepwise_returns: np.ndarray, log_prob_actions: np.ndarray) -> float:
    """Calculate policy gradient loss (numpy version, no torch dependency)"""
    if len(stepwise_returns) == 0 or len(log_prob_actions) == 0:
        return 0.0
    loss = -(stepwise_returns * log_prob_actions).sum()
    return float(loss)


def train_q_learning(env: MedicalEnvironment, agent: QLearningAgent, 
                    episodes: int = 100000) -> dict:
    """Train Q-learning agent"""
    
    # Metrics tracking
    episode_rewards = []
    episode_accuracies = []
    cumulative_rewards = []
    success_rate_history = []
    
    # Confusion matrix tracking
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    print("Starting Q-Learning Training...")
    
    for episode in range(1, episodes + 1):
        try:
            state = env.reset()
            total_reward = 0
            epochs = 0
            
            while not env.done:
                # Choose action
                action = agent.choose_action(state)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Update Q-value
                agent.update_q_value(state, action, reward, next_state)
                
                # Update metrics
                total_reward += reward
                
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
                epochs += 1
                
                # Optional rendering for first few episodes
                if episode <= 5:
                    env.render()
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Track metrics
            episode_rewards.append(total_reward)
            accuracy = (true_positives + true_negatives) / max(1, episode)
            episode_accuracies.append(accuracy)
            cumulative_rewards.append(sum(episode_rewards))
            
            # Calculate success rate (avoiding unnecessary interventions)
            total_predictions = true_positives + true_negatives + false_positives + false_negatives
            success_rate = (true_positives + true_negatives) / max(1, total_predictions)
            success_rate_history.append(success_rate)
            
            # Print progress
            if episode % 10000 == 0:
                avg_reward = np.mean(episode_rewards[-1000:]) if len(episode_rewards) >= 1000 else np.mean(episode_rewards)
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Accuracy = {accuracy:.3f}, Success Rate = {success_rate:.3f}, "
                      f"Epsilon = {agent.epsilon:.3f}")
                      
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


def evaluate_agent(env: MedicalEnvironment, agent: QLearningAgent, 
                  test_episodes: int = 1000) -> dict:
    """Evaluate trained agent"""
    
    test_rewards = []
    predictions = []
    true_labels = []
    
    # Temporarily set epsilon to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(test_episodes):
        try:
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                total_reward += reward
                predictions.append(action)
                true_labels.append(info['true_action'])
                
                state = next_state
            
            test_rewards.append(total_reward)
            
        except Exception as e:
            print(f"Error in evaluation episode {episode}: {e}")
            continue
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    if not predictions:
        print("Warning: No successful evaluation episodes")
        return {
            'accuracy': 0, 'average_reward': 0, 'precision': 0, 'recall': 0, 
            'specificity': 0, 'confusion_matrix': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        }
    
    # Calculate evaluation metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    accuracy = np.mean(predictions == true_labels)
    avg_reward = np.mean(test_rewards)
    
    # Confusion matrix
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
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    }


def plot_training_metrics(metrics: dict, save_path: str = None):
    """Plot training metrics"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative Reward
        if metrics['cumulative_rewards']:
            axes[0, 0].plot(metrics['cumulative_rewards'])
            axes[0, 0].set_title('Cumulative Reward')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Cumulative Reward')
            axes[0, 0].grid(True)
        
        # Success Rate
        if metrics['success_rate_history']:
            axes[0, 1].plot(metrics['success_rate_history'])
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True)
        
        # Average Reward (rolling window)
        window_size = min(1000, len(metrics['episode_rewards']) // 10)
        if len(metrics['episode_rewards']) >= window_size and window_size > 0:
            rolling_rewards = pd.Series(metrics['episode_rewards']).rolling(window=window_size).mean()
            axes[1, 0].plot(rolling_rewards)
            axes[1, 0].set_title(f'Average Reward (Rolling {window_size})')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Average Reward')
            axes[1, 0].grid(True)
        
        # Sample Efficiency (Accuracy over time)
        if metrics['episode_accuracies']:
            axes[1, 1].plot(metrics['episode_accuracies'])
            axes[1, 1].set_title('Sample Efficiency (Accuracy)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training metrics plot saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting training metrics: {e}")


def plot_confusion_matrix(confusion_matrix: dict, title: str = "Confusion Matrix"):
    """Plot confusion matrix"""
    
    try:
        cm = np.array([[confusion_matrix['TN'], confusion_matrix['FP']],
                       [confusion_matrix['FN'], confusion_matrix['TP']]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Monitor', 'Medicate'],
                    yticklabels=['Monitor', 'Medicate'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")


def main():
    """Main function to run Q-learning for medical monitoring"""
    
    try:
        # Configuration
        data_path = "/Users/lukman/Desktop/Research/medical_dataset.csv"
        n_states = 100
        n_episodes = 10000  # Reduced for initial testing
        
        # Hyperparameters (matching your pseudo-code)
        alpha = 0.1      # Learning rate
        gamma = 0.6      # Discount factor  
        epsilon = 0.1    # Exploration rate
        
        print("Initializing Medical Environment...")
        
        # Create environment
        env = MedicalEnvironment(data_path, n_states=n_states)
        
        print(f"Environment created with {env.observation_space.n} states and {env.action_space.n} actions")
        print(f"Dataset shape: {env.df.shape}")
        print(f"Features used: {env.feature_columns}")
        
        # Create Q-learning agent
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
        
        print(f"Q-table initialized with shape: {agent.Q.shape}")
        
        # Train agent
        print("\nStarting training...")
        training_metrics = train_q_learning(env, agent, episodes=n_episodes)
        
        print("\nTraining completed!")
        print(f"Final Accuracy: {training_metrics['final_accuracy']:.3f}")
        print(f"Final Success Rate: {training_metrics['final_success_rate']:.3f}")
        
        # Evaluate agent
        print("\nEvaluating agent...")
        eval_metrics = evaluate_agent(env, agent, test_episodes=1000)
        
        print("\nEvaluation Results:")
        print(f"Accuracy: {eval_metrics['accuracy']:.3f}")
        print(f"Average Reward: {eval_metrics['average_reward']:.2f}")
        print(f"Precision: {eval_metrics['precision']:.3f}")
        print(f"Recall: {eval_metrics['recall']:.3f}")
        print(f"Specificity: {eval_metrics['specificity']:.3f}")
        
        # Plot results
        print("\nGenerating plots...")
        plot_training_metrics(training_metrics, save_path="q_learning_training_metrics.png")
        plot_confusion_matrix(eval_metrics['confusion_matrix'], "Evaluation Confusion Matrix")
        
        # Save Q-table and metrics
        np.save("q_table_medical.npy", agent.Q)
        print("Q-table saved to: q_table_medical.npy")
        
        # Save metrics to CSV
        if training_metrics['episode_rewards']:
            metrics_df = pd.DataFrame({
                'episode': range(1, len(training_metrics['episode_rewards']) + 1),
                'reward': training_metrics['episode_rewards'],
                'accuracy': training_metrics['episode_accuracies'],
                'cumulative_reward': training_metrics['cumulative_rewards'],
                'success_rate': training_metrics['success_rate_history']
            })
            metrics_df.to_csv("q_learning_metrics.csv", index=False)
            print("Metrics saved to: q_learning_metrics.csv")
        
        print("\nTraining and evaluation completed successfully!")
        
        return agent, env, training_metrics, eval_metrics
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    agent, env, training_metrics, eval_metrics = main()