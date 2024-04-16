import argparse
import torch


# PARSE ARGUMENTS
def get_args():
    parser = argparse.ArgumentParser(description="VAE for missing value imputation.")

    parser.add_argument("--f")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device", default=device, help="Set the device.")


    parser.add_argument("--MODEL_OPTIONS", default="------------------------")
    parser.add_argument(
        "--dataset_name",
        default="ortho", # wine, boston, covtype, ortho
        type=str,
        help="Set the dataset name.",
    )
    parser.add_argument(
        "--missing_pattern", # single, multiple, random
        default="single",
        type=str,
        help="Set the missing pattern.",
    )
    parser.add_argument(
        "--col_to_remove", # only used in SINGLE missing pattern
        default=3,
        type=int,
        help="Set the column to remove.",
    )
    parser.add_argument(
        "--include_complete",
        default=True,
        type=bool,
        help="Set whether to include complete data.",
    )
    parser.add_argument(
        # used in MULTIPLE and RANDOM missing pattern       
        "--max_remove_count", 
        default=2,
        type=int,
        help="Set how many features can be removed.",
    )
    parser.add_argument(
        # only used in RANDOM missing pattern
        "--new_num_per_origin",
        default=500,
        type=int,
        help="Set how many new rows you want to create per origin row.",
    )
    parser.add_argument(
        "--kld_rate",
        default=0.01,
        type=float,
        help="Set the KLD rate.",
    )


    parser.add_argument("--LAYER_DIMENSIONS", default="-----------------------")
    parser.add_argument("--H1", default=128, help="Set the first hidden dimension.")
    parser.add_argument("--H2", default=64, help="Set the second hidden dimension.")
    parser.add_argument("--H3", default=32, help="Set the second hidden dimension.")
    parser.add_argument("--latent_dim", default=21, help="Set the latent dimension.")


    parser.add_argument("--IMPLEMENT_SETTINGS", default="----------------------")
    parser.add_argument(
        "--epochs", default=15000, type=int, help="Set the number of epochs."
    )
    parser.add_argument("--batch_size", default=1024, help="Set the batch size.")
    parser.add_argument(
        "--val_rate",
        default=0.2,
        type=float,
        help="Set the validation rate.",
    )
    parser.add_argument(
        "--test_rate",
        default=0.1,
        type=float,
        help="Set the test rate.",
    )
    parser.add_argument(
        "--print_period",
        default=10,
        type=int,
        help="Set the print period.",
    )


    parser.add_argument("--SCHEDULER_SETTINGS", default="----------------------")
    parser.add_argument(
        "--learning_rate",
        default=1e-2,
        type=float,
        help="Set the step size for the scheduler.",
    )
    parser.add_argument(
        "--step_size",
        default=500,
        type=int,
        help="Set the step size for the scheduler.",
    )
    parser.add_argument(
        "--gamma",
        default=0.8,
        type=float,
        help="Set the gamma for the scheduler.",
    )
    args = parser.parse_args()

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

    return args
