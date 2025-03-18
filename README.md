# Bottle Anomaly Detection Pipeline

```mermaid
flowchart TD
    A[Start] --> B[Extract Frames from Video]
    B --> C[Detect and Crop Bottles]
    C --> D[Create Synthetic Anomalies]
    
    D --> E1[Add Synthetic Cracks]
    D --> E2[Add Synthetic Dirt]
    D --> E3[Add Synthetic Air Bubbles]
    
    E1 --> F[Apply Data Augmentation]
    E2 --> F
    E3 --> F
    
    F --> G[Save and Label Dataset]
    G --> H[Dataset Ready for Training]
    
    H --> I[YOLO Object Detection]
    I --> J[Unsupervised Anomaly Learning]
    J --> K[Model Evaluation]
    K --> L[Deployment]
    L --> M[End]
    
    classDef default fill:#2C3E50,stroke:#7FB3D5,stroke-width:2px,color:#ECF0F1,font-weight:bold;
    classDef highlight fill:#3498DB,stroke:#2980B9,stroke-width:2px,color:#ECF0F1;
    classDef anomaly fill:#E74C3C,stroke:#C0392B,stroke-width:2px,color:#ECF0F1;
    
    class A,M default;
    class D,E1,E2,E3,F highlight;
    class I,J,K anomaly;
