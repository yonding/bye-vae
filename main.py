from get_args import get_args
from get_vae import get_vae
from load_datasets import load_datasets
from get_dataloaders import get_dataloaders
import torch.optim as optim
from vae_loss import customLoss
from torch.optim.lr_scheduler import StepLR
import torch

def main(args):
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=float(10000))
    X_miss_train, Z_miss_train, y_miss_train, X_miss_val, Z_miss_val, y_miss_val, X_miss_test, Z_miss_test, y_miss_test = load_datasets(args)
    train_loader, val_loader, test_loader = get_dataloaders(args, X_miss_train, Z_miss_train, X_miss_val, Z_miss_val, X_miss_test, Z_miss_test)
    model = get_vae(args)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = customLoss(args)  # MSE + KLD

    # TRAINING
    print("\n\n\n")
    print("================================================================================")
    print("=============================== STRART TRAINING ================================")
    print("================================================================================")
    
    # 얼리 스톱을 위한 변수 설정
    prev_val_loss = float('inf')
    increasing_loss_count = 0

    # 가장 작은 val_mse_loss와 그에 해당하는 값들을 저장하기 위한 변수 설정
    min_val_mse_loss = float('inf')
    best_epoch = 0
    best_train_loss = 0
    best_train_mse_loss = 0
    best_val_loss = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mse_loss = train(args, epoch, model, train_loader, optimizer, scheduler, loss_fn)

        if epoch % args.print_period == 0:
            val_loss, val_mse_loss = valid(epoch, model, val_loader, loss_fn, args)
            print("====> Epoch: {} Train loss: {:.4f} Train nRMSE: {:.4f} Val loss: {:.4f} Valid nRMSE: {:.4f}".format(epoch, train_loss, train_mse_loss ** 0.5, val_loss, val_mse_loss ** 0.5))
            print("")
            print("")

            # val_mse_loss가 가장 작은 에포크와 그에 해당하는 값들 저장
            if val_mse_loss < min_val_mse_loss:
                min_val_mse_loss = val_mse_loss
                best_epoch = epoch
                best_train_loss = train_loss
                best_train_mse_loss = train_mse_loss
                best_val_loss = val_loss

            # 검증 손실이 증가하면 카운트 증가
            if val_loss > prev_val_loss:
                increasing_loss_count += 1
            else:
                increasing_loss_count = 0

            # 검증 손실이 5번 연속으로 증가하면 학습 중단
            if increasing_loss_count >= 100:
                print("Early stopping due to validation loss increasing for 5 consecutive epochs.")
                break

            prev_val_loss = val_loss

    # 가장 작은 val_mse_loss와 그에 해당하는 에포크와 손실 값들 출력
    print("Best Epoch: {} Train loss: {:.4f} Train nRMSE: {:.4f} Val loss: {:.4f} Valid nRMSE: {:.4f}".format(best_epoch, best_train_loss, best_train_mse_loss ** 0.5, best_val_loss, min_val_mse_loss ** 0.5))

    print("----------------------------------------------")
    checkpoint_path = f"./main_{args.max_remove_count}_{args.new_num_per_origin}_{args.epochs}_{args.H1}_{args.H2}_{args.latent_dim}.pth"

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)

def train(
    args, epoch, model, train_loader, optimizer, scheduler, loss_fn
):
    model.train()
    train_loss = 0
    mse_loss = [0]
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_fn(recon_batch, targets, mu, logvar, mse_loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()

    if epoch % args.print_period == 0:
        print("")
        print("")
        print("--------------------TRAIN RESULT-----------------------")   
        for d, r, t in zip(data[:2], recon_batch[:2], targets[:2]):
            print("I: ", d)
            print("R: ", r)
            print("G: ", t)
            print("-------------------------------------------------------")
    
    return train_loss / len(train_loader.dataset), mse_loss[0] / len(train_loader.dataset)


def valid(epoch, model, val_loader, loss_fn, args):
    model.eval()
    val_loss = 0
    mse_loss = [0]
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            data = data.to(args.device)
            targets = targets.to(args.device)

            recon_batch, mu, logvar = model(data)
            loss = loss_fn(recon_batch, targets, mu, logvar, mse_loss)
            val_loss += loss.item()
    
    if epoch % args.print_period == 0:
        print()
        print("--------------------VALID RESULT-----------------------")   
        for d, r, t in zip(data[:2], recon_batch[:2], targets[:2]):
            print("I: ", d)
            print("R: ", r)
            print("G: ", t)
            print("-------------------------------------------------------")

    return val_loss / len(val_loader.dataset), mse_loss[0] / len(val_loader.dataset)


if __name__ == "__main__":
    args = get_args()
    main(args)
    print()
    print()
    print("----------------- SETTINGS -------------------")
    not_to_print = {'single':['f', 'max_remove_count', 'new_num_per_origin'], 'multiple':['f', 'new_num_per_origin', 'col_to_remove'], 'random':['f', 'col_to_remove']}
    
    max_length = max(len(arg) for arg in vars(args))
    for arg, value in vars(args).items():
        if arg in ['SCHEDULER_SETTINGS', 'MODEL_OPTIONS', 'LAYER_DIMENSIONS', 'IMPLEMENT_SETTINGS']:
            print()
            print("["+arg+"]")
            continue
        if arg not in not_to_print[args.missing_pattern]:
            print(f"{arg.ljust(max_length)}: {value}")
    print("----------------------------------------------")
    print()


