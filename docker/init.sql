-- Executado pelo entrypoint do Postgres apenas na primeira criação do volume.
-- Habilita o tipo `vector` e os operadores de distância do pgvector.
CREATE EXTENSION IF NOT EXISTS vector;
