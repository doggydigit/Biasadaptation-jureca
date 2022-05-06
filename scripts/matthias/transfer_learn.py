import sys
import pickle
from pathlib import Path
from os.path import isfile
from data_helper import get_all_data
from train_helper import all_seeds
from train_full_dataset import transfer_bmr_full_dataset


# def train_transfer(nr_hidden_neurons, train_data, validation_data, transfertrain_data, transfervalidation_data,
#                    highseed, trainbw_result_dir, transferb_result_dir, verbose=False, debug=False):
#
#     # Initilialize parameters
#     if debug:
#         nr_epochs = 2
#         batch_size = 180
#         seeds = range(2)
#     else:
#         nr_epochs = 50
#         batch_size = 200
#         seeds = range(highseed)
#     bw_lr = 0.001
#     b_lr = 0.01
#     loss_function = "mse"
#     readout_function = "hardtanh"
#     result_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
#
#     # First train weights then transfer learn biases
#     for transferlearning in [False, True]:
#
#         # Initialize directories and paths to save simulation results
#         if transferlearning:
#             print("\n\nTransfer learning phase started:\n")
#             result_dir = transferb_result_dir
#             result_path = result_dir + "/network_{}_{}_{}_blr_{}.pickle".format(nr_hidden_neurons, loss_function,
#                                                                                 readout_function, b_lr)
#         else:
#             print("\n\nWeight training phase started:\n")
#             result_dir = trainbw_result_dir
#             result_path = result_dir + "/network_{}_{}_{}_bwlr_{}.pickle".format(nr_hidden_neurons, loss_function,
#                                                                                  readout_function, bw_lr)
#         weight_save_dir = result_dir + "/weights"
#         Path(result_dir).mkdir(parents=True, exist_ok=True)
#         Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
#
#         # Object to store simulation results
#         results = {"loss_function": loss_function,
#                    "readout_function": readout_function,
#                    "bw_learning_rate": bw_lr,
#                    "seeds": list(seeds)}
#         for r in result_types:
#             results[r] = []
#
#         # Repeat simulations for different random seeds
#         for seed in seeds:
#             all_seeds(seed)
#             if verbose:
#                 print("Training  network {} with loss {}, readout {}, learning rate {} and seed {}"
#                       "".format(nr_hidden_neurons, loss_function, readout_function, bw_lr, seed))
#
#             # Define path where trained model will be saved
#             if debug:
#                 weight_save_path = None
#             else:
#                 if transferlearning:
#                     weight_save_path = weight_save_dir + "/network_{}_lossf_{}_readoutf_{}_blr_{}_seed_{}.pickle" \
#                                                          "".format(nr_hidden_neurons, loss_function, readout_function,
#                                                                    b_lr, seed)
#                 else:
#                     weight_save_path = weight_save_dir + "/network_{}_lossf_{}_readoutf_{}_bwlr_{}_seed_{}.pickle" \
#                                                          "".format(nr_hidden_neurons, loss_function, readout_function,
#                                                                    bw_lr, seed)
#
#             # Perform simulation or load results in case it was already performed
#             if debug or not isfile(weight_save_path):
#                 if transferlearning:
#                     weight_load_path = trainbw_result_dir + "/weights/network_{}_lossf_{}_readoutf_{}_bwlr_{}_seed_{}" \
#                                                             ".pickle".format(nr_hidden_neurons, loss_function,
#                                                                              readout_function, bw_lr, seed)
#                     train_performance, train_loss, validation_performance, validation_loss = train_eval(
#                         nr_hidden_neurons, transfertrain_data, transfervalidation_data, b_lr=b_lr, nr_epochs=nr_epochs,
#                         batch_size=batch_size, loss_function=loss_function, readout_function=readout_function,
#                         weight_save_path=weight_save_path, weight_load_path=weight_load_path, verbose=verbose
#                     )
#                 else:
#                     train_performance, train_loss, validation_performance, validation_loss = train_eval(
#                         nr_hidden_neurons, train_data, validation_data, bw_lr=bw_lr, nr_epochs=nr_epochs,
#                         batch_size=batch_size, loss_function=loss_function, readout_function=readout_function,
#                         weight_save_path=weight_save_path, verbose=verbose
#                     )
#             else:
#                 if isfile(result_path):
#                     with open(result_path, "rb") as f:
#                         load_results = pickle.load(f)
#                         for r in result_types:
#                             results[r] += [load_results[r][seed]]
#                         train_performance = load_results["train_performance"][seed]
#                         train_loss = load_results["train_loss"][seed]
#                         validation_performance = load_results["validation_performance"][seed]
#                         validation_loss = load_results["validation_loss"][seed]
#                     print("Seed {} simulation was already performed with training performance {}, validation "
#                           "performance {} and was therefore loaded from save"
#                           "".format(seed, train_performance, validation_performance))
#                 else:
#                     raise FileExistsError("Although the weights have already been trained, the simulation results "
#                                           "can't be found.")
#
#             # Store simulation outcomes
#             results["train_performance"] += [train_performance]
#             results["train_loss"] += [train_loss]
#             results["validation_performance"] += [validation_performance]
#             results["validation_loss"] += [validation_loss]
#             if verbose:
#                 print("Found training performance {:.1f}%, training loss {} and validation performance "
#                       "{:.1f}".format(train_performance, train_loss, validation_performance))
#
#         # Print average performance
#         average_performance = sum(results["validation_performance"]) / len(seeds)
#         print("Network {} had validation performance of {}%".format(nr_hidden_neurons, average_performance))
#
#         # Save outcome of training to pickle file
#         if not debug:
#             with open(result_path, 'wb') as handle:
#                 pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# def train_letters_transfer_mnist(nr_hidden_neurons, highseed=25, verbose=False, debug=False, root_dir="../../"):
#
#     # Get EMNIST_letters training data to train weights
#     traindata, validation_data = get_all_data("EMNIST_letters", debug=debug)
#
#     # Get MNIST training data to transfer learn with biases
#     transfertrain_data, transfervalidation_data = get_all_data("MNIST", debug=debug)
#
#     # Define directories to save simulation results
#     trainbw_result_dir = root_dir + "results/trainbw/EMNIST_letters"
#     transferb_result_dir = root_dir + "results/train_letters_transfer_mnist"
#
#     # Simulate weight training and transfer learning
#     train_transfer(nr_hidden_neurons, traindata, validation_data, transfertrain_data, transfervalidation_data,
#                    highseed, trainbw_result_dir, transferb_result_dir, verbose, debug)
#
#
# def train_mnist_transfer_letters(nr_hidden_neurons, highseed=25, verbose=False, debug=False, root_dir="../../"):
#
#     # Get EMNIST_letters training data to train weights
#     traindata, validation_data = get_all_data("MNIST", debug=debug)
#
#     # Get MNIST training data to transfer learn with biases
#     transfertrain_data, transfervalidation_data = get_all_data("EMNIST_letters", debug=debug)
#
#     # Define directories to save simulation results
#     trainbw_result_dir = root_dir + "results/trainbw/MNIST"
#     transferb_result_dir = root_dir + "results/train_mnist_transfer_letters"
#
#     # Simulate weight training and transfer learning
#     train_transfer(nr_hidden_neurons, traindata, validation_data, transfertrain_data, transfervalidation_data,
#                    highseed, trainbw_result_dir, transferb_result_dir, verbose, debug)
#
#
# def train_emnist_transfer_mnist(nr_hidden_neurons, highseed=25, verbose=False, debug=False, root_dir="../../"):
#
#     # Get EMNIST_letters training data to train weights
#     traindata, validation_data = get_all_data("EMNIST", debug=debug)
#
#     # Get MNIST training data to transfer learn with biases
#     transfertrain_data, transfervalidation_data = get_all_data("MNIST", debug=debug)
#
#     # Define directories to save simulation results
#     trainbw_result_dir = root_dir + "results/trainbw/EMNIST"
#     transferb_result_dir = root_dir + "results/train_emnist_transfer_mnist"
#
#     # Simulate weight training and transfer learning
#     train_transfer(nr_hidden_neurons, traindata, validation_data, transfertrain_data, transfervalidation_data,
#                    highseed, trainbw_result_dir, transferb_result_dir, verbose, debug)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        transfer_bmr_full_dataset(nrhidden=[10], transfer_dataset="K49", load_weight_types="EMNIST_bymerge",
                                  recompute=True, verbose=True, save_performance=False)
    elif sys.argv[1] == "transfer_k49_from_emnist":
        net = list(map(int, sys.argv[2].split(',')))
        transfer_bmr_full_dataset(nrhidden=net, transfer_dataset="K49", load_weight_types="EMNIST_bymerge",
                                  recompute=True, verbose=True, save_performance=True)
    else:
        raise ValueError(sys.argv)
