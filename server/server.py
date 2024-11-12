# server/server.py

import flwr as fl

def main():
    # Define the strategy: Federated Averaging (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=4,        
        min_evaluate_clients=4,   
        min_available_clients=4,  
    )
    
    # Start the Flower server
    fl.server.start_server(
        server_address="localhost:8081",
        config=fl.server.ServerConfig(num_rounds=50),  # Adjust number of rounds as needed
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
