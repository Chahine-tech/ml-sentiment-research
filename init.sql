-- Suppression des tables si elles existent déjà
DROP TABLE IF EXISTS comments;
DROP TABLE IF EXISTS model_metrics;

-- Table pour stocker les commentaires
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    cleaned_text TEXT,
    hate_label INTEGER,
    prediction INTEGER,
    confidence DOUBLE PRECISION,
    model_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table pour stocker les métriques des modèles
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    accuracy DOUBLE PRECISION NOT NULL,
    precision DOUBLE PRECISION NOT NULL,
    recall DOUBLE PRECISION NOT NULL,
    f1_score DOUBLE PRECISION NOT NULL,
    model_parameters JSONB,
    model_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ajout d'index pour améliorer les performances des requêtes
CREATE INDEX comments_model_type_idx ON comments(model_type);
CREATE INDEX model_metrics_model_type_idx ON model_metrics(model_type);

-- Message de confirmation
SELECT 'Tables initialization completed successfully!' as message; 