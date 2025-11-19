graph TD
    %% --- DEFINICIÓN DE ESTILOS ---
    classDef energy fill:#f96,stroke:#333,stroke-width:2px;
    classDef skin fill:#ffcccc,stroke:#f00,stroke-width:2px;
    classDef coreSafe fill:#ccffcc,stroke:#0f0,stroke-width:2px;
    classDef coreLocked fill:#ff9999,stroke:#f00,stroke-width:4px;
    classDef layer fill:#fff,stroke:#333,stroke-dasharray: 5 5;

    %% --- BLOQUE BINARIO ---
    subgraph "SC_BINARY (Ej. MgB2)"
        B_Input(Tensión Estructural ξ) -->|Impacto Alto| B_Skin[Piel p=2<br/>Alta Disipación]
        B_Skin -->|Decaimiento Rápido| B_Mid[Capas Medias p=3,5]
        B_Mid -.->|Residuo Nulo| B_Core[Núcleo p=7<br/>ESTABLE / LIMPIO]
    end

    %% --- BLOQUE BASE HIERRO ---
    subgraph "SC_IRON-BASED (Ej. FeSe)"
        I_Input(Tensión Estructural ξ) -->|Impacto Moderado| I_Skin[Piel p=2<br/>Baja Disipación]
        I_Skin -->|Transmisión| I_Mid[Capas Medias p=3,5]
        I_Mid ==>|Acoplamiento Fuerte| I_Core[Núcleo p=7<br/>ACOPLADO / ACTIVO]
        I_Core -.->|Feedback de Ruido| I_Skin
    end

    %% --- ASIGNACIÓN DE CLASES ---
    class B_Input,I_Input energy;
    class B_Skin skin;
    class B_Core coreSafe;
    class I_Core coreLocked;
    class B_Mid,I_Mid layer;