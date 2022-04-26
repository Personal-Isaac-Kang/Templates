"""
Main train function.

- Build model.
- Configure dataloaders.
- Configure optimizers.
- Train with trainer.

"""

from funcs.engine.trainer import Trainer

def main():
    model = build_model()
    train_dataloader = get_dataloader('train')
    val_dataloader = get_dataloader('val')
    optimizer = optimizer
    
    trainer = Trainer()
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()