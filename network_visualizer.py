"""
INTERACTIVE NETWORK VISUALIZATION
Creates beautiful 3D network graphs of fraud connections
"""

import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def create_3d_fraud_network(num_nodes=200, fraud_ratio=0.035):
    """Create 3D interactive network visualization"""
    
    # Generate network
    G = nx.barabasi_albert_graph(num_nodes, 3, seed=42)
    
    # Assign fraud labels
    fraud_labels = np.random.choice([0, 1], size=num_nodes, 
                                   p=[1-fraud_ratio, fraud_ratio])
    
    # 3D layout
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Extract coordinates
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]
    
    colors = ['red' if fraud_labels[n] == 1 else 'lightblue' for n in G.nodes()]
    sizes = [15 if fraud_labels[n] == 1 else 8 for n in G.nodes()]
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='lightgray', width=1),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color='white', width=0.5)
        ),
        text=[f"Transaction {n}<br>{'FRAUD' if fraud_labels[n]==1 else 'LEGITIMATE'}<br>Connections: {G.degree[n]}" 
              for n in G.nodes()],
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"3D Transaction Network<br>Red = Fraud ({fraud_labels.sum()} nodes) | Blue = Legitimate ({(fraud_labels==0).sum()} nodes)",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
        ),
        showlegend=False,
        height=800
    )
    
    return fig

if __name__ == "__main__":
    fig = create_3d_fraud_network(num_nodes=300)
    fig.write_html("fraud_network_3d.html")
    print("✅ Saved to 'fraud_network_3d.html' - Open in browser!")
    fig.show()