$env:PYTHONPATH="code"
python -m tools.generate_all_visuals `
  --model .\outputs\zerogs_train\checkpoints\latest.pt `
  --image .\dataset\renders_256