Aqui vai o que o experimento produz ao final (outputs) e onde encontrar:

- Checkpoints do modelo
  - Arquivos: checkpoint.pt (último) e best{target_domain}.checkpoint.pt (melhor)
  - Local: em config.logging.path (no seu setup, yaml a partir do CWD).
- Métricas de desempenho
  - Impressas no console: “avg performance”, “test avg performance”, “best performance”, “best att performance”, além de “CLS Performance” (QWK global) e QWK por atributo.
  - Não são gravadas em arquivo por padrão.
- Cópia da configuração usada
  - Arquivo: config.yaml
  - Local: config.yaml (mesmo diretório de logs/checkpoints).
- Gráficos t‑SNE (diagnóstico)
  - Arquivos: `../img/{target_domain}epoch{N}clsAtt.svg` (ex.: 5epoch0clsAtt.svg)
  - Observação: o caminho começa com `../`, então salva uma pasta acima do diretório de execução.
- Dados do dataset (pré-processados)
  - Arquivos: `datasets/AES/{1..8}/{train,dev,test}.pk` (já gerados pelo script).

Nota: Não há CSV/JSON de predições salvo automaticamente. Se quiser, posso adicionar a gravação de métricas e predições (ex.: CSV na pasta de logs) e ajustar o path dos gráficos para dentro do repositório.
