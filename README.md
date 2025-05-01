# Federated Learning Simulation

This repository contains code to simulate **Federated Learning** on a single machine.

---

## ⚙️ Configuration

You can customize the simulation by modifying the following variables in the code:

- **`K`** — Total number of clients participating in the simulation.
- **`N`** — Number of *active clients* selected in each global round.
- **`frac`** — A list that defines the number of training samples assigned to each client.  

This means:

- The first 10 clients each get 1000 training samples.

- The next 10 clients each get 2000 training samples.

- out_epoch — Number of global rounds (i.e., the number of times the server aggregates updates from clients).

- in_epoch — Number of local training epochs each client performs on their data before communicating with the server.

## 💡 Purpose

This simulation is ideal for experimenting with different federated learning configurations and understanding the impact of:

- Varying client data distributions

- Partial client participation

- Local vs global training dynamics
