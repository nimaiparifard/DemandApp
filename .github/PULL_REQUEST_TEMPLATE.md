## What and why

## How it was tested
- [ ] `make test`
- [ ] `make helm-lint` / `make tofu-check` (if infra changed)
- [ ] Model change: backtest table attached (WRMSSE / MAE / bias vs. current champion)

## Risk
Does this change feature definitions? (then training AND serving need a new model version)
