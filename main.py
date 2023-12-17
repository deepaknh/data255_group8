import argparse
import os
from preprocessing import DataPreprocessor
from bigru_model import BiGRU_model
from lstm_model import LSTM_model
from gru_model import GRU_model
from bilstm_model import BiLSTM_model
from download_data import Download_Data
import pandas as pd

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Train or test a MBTI classification model')
    parser.add_argument("--model_type", help="Type of the MBTI model", choices=["EI", "SN", "TF", "JP"])
    parser.add_argument("--model_architecture", help="Architecture of the MBTI model", choices=["LSTM", "BiLSTM", "GRU", "BiGRU"])
    parser.add_argument("--download_data", help="Download necessary data", action='store_true')

    args = parser.parse_args()

    if args.download_data:
        # Download necessary data
        print("Training and Testing datasets are being downloaded from the drive...")
        download_data = Download_Data()
        download_data.download_training_data()
        download_data.download_testing_data()
        print("Training and Testing datasets download complete...")

    # Load your data
    data = pd.read_csv('TRAINING.csv')

    # Split type
    data_split = data.copy()
    data_split[["EI", "SN", "TF", "JP"]] = data["type"].apply(list).tolist()

    # Preprocess your data
    preprocessor = DataPreprocessor(data_split)

    # Split dataset for each MBTI dimension
    X_train, X_val, y_train, y_val = preprocessor.split_dataset(args.model_type)
    print("Data Splitting into Training and Validation Done...")

    # Oversample training data
    X_train, y_train = preprocessor.oversample_data(X_train, y_train)
    print("Oversampling Training data Done...")

    # Tokenize and pad training data
    X_padded_train, tokenizer = preprocessor.tokenize_and_pad(X_train)
    print("Tokenizng and Padding Training data Done...")

    # Tokenize and pad validation data
    X_padded_val, _ = preprocessor.tokenize_and_pad(X_val, tokenizer=tokenizer)
    print("Tokenizng and Padding Validation data Done...")

    # Encode the labels
    y_train_encoded, label_encoder = preprocessor.label_encoding(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    print("Label Encoding of Training and Testing data Done...")

    # Train and test based on the chosen architecture
    if args.model_architecture == 'LSTM':
        print("Entering training...")
        lstm_model = LSTM_model(input_dim=10000, output_dim=1, input_length=2000)
        # Build and train the model
        model = lstm_model.create_deep_model()
        history = lstm_model.train_and_monitor_model(model, X_padded_train, y_train_encoded, X_padded_val, y_val_encoded)

        # Plot the training history
        lstm_model.plot_history(history, f'LSTM Classification - {args.model_type}')

        print("Entering testing...")
        
        test_data = pd.read_csv('TESTING.csv')
        test_data[["EI", "SN", "TF", "JP"]] = test_data["type"].apply(list).tolist()
        X_test = test_data['posts']
        y_test = test_data[args.model_type]
        X_padded_test, _ = preprocessor.tokenize_and_pad(X_test, tokenizer=tokenizer)

        # Test the model
        lstm_model.test_model(X_padded_test, y_test,args.model_type)

    elif args.model_architecture == 'BiGRU':
        bigru_model = BiGRU_model(input_dim=10000, output_dim=1, input_length=2000)
        # Build and train the model
        model = bigru_model.create_deep_model()
        history = bigru_model.train_and_monitor_model(model, X_padded_train, y_train_encoded, X_padded_val, y_val_encoded)

        # Plot the training history
        bigru_model.plot_history(history, f'BiGRU Classification - {args.model_type}')

        
        test_data = pd.read_csv('TESTING.csv')
        test_data[["EI", "SN", "TF", "JP"]] = test_data["type"].apply(list).tolist()
        X_test = test_data['posts']
        y_test = test_data[args.model_type]
        X_padded_test, _ = preprocessor.tokenize_and_pad(X_test, tokenizer=tokenizer)

        # Test the model
        bigru_model.test_model(X_padded_test, y_test,args.model_type)

    elif args.model_architecture == 'BiLSTM':
        bilstm_model = BiLSTM_model(input_dim=10000, output_dim=1, input_length=2000)
        # Build and train the model
        model = bilstm_model.create_deep_model()
        history = bilstm_model.train_and_monitor_model(model, X_padded_train, y_train_encoded, X_padded_val, y_val_encoded)

        # Plot the training history
        bilstm_model.plot_history(history, f'BiLSTM Classification - {args.model_type}')

        
        test_data = pd.read_csv('TESTING.csv')
        test_data[["EI", "SN", "TF", "JP"]] = test_data["type"].apply(list).tolist()
        X_test = test_data['posts']
        y_test = test_data[args.model_type]
        X_padded_test, _ = preprocessor.tokenize_and_pad(X_test, tokenizer=tokenizer)

        # Test the model
        bilstm_model.test_model(X_padded_test, y_test,args.model_type)

    elif args.model_architecture == 'GRU':
        gru_model = GRU_model(input_dim=10000, output_dim=1, input_length=2000)
        # Build and train the model
        model = gru_model.create_deep_model()
        history = gru_model.train_and_monitor_model(model, X_padded_train, y_train_encoded, X_padded_val, y_val_encoded)

        # Plot the training history
        gru_model.plot_history(history, f'GRU Classification - {args.model_type}')

        
        test_data = pd.read_csv('TESTING.csv')
        test_data[["EI", "SN", "TF", "JP"]] = test_data["type"].apply(list).tolist()
        X_test = test_data['posts']
        y_test = test_data[args.model_type]
        X_padded_test, _ = preprocessor.tokenize_and_pad(X_test, tokenizer=tokenizer)

        # Test the model
        gru_model.test_model(X_padded_test, y_test,args.model_type)
