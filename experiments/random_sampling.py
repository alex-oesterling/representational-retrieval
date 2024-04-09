from representational_retrieval import *
import torch
import itertools
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import argparse
import seaborn as sns
sns.set_style("whitegrid")

def sup_over_trees(dataloader, attributes, k=10, depth=2):
    print(functions)
    retrieved_sums = torch.zeros(len(functions))
    for index, function in enumerate(functions):
        retrieved_sums[index] = torch.sum(sample[:, list(function)]) ## Here I am doing a sum operation for vector-valued functions c (such as "gender" and "race"). We can change to norm or any other aggregation method.
    
    retrieved_sums /= k

    print(torch.max(torch.abs(retrieved_sums-sums)))
    return torch.max(torch.abs(retrieved_sums-sums))

def generate_tree(depth, attributes):
    functions = []
    for i in range(1, depth+1):
        functions.extend(list(itertools.combinations(list(range(len(attributes))), i)))
    return functions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-depth', type=int)
    args = parser.parse_args()
    depth = args.depth

    attributes = [
        'Male',
        'Eyeglasses',
        'Smiling',
        'Young'
    ]

    dataset = CelebA("/n/holylabs/LABS/hlakkaraju_lab/Lab/datasets/", attributes=attributes, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    functions = generate_tree(depth, attributes)

    ## compute aggregate statistics
    expected_function_values = torch.zeros(len(functions))
    total = 0
    for _, labels in tqdm(dataloader):
        for index, function in enumerate(functions):
            expected_function_values[index] = torch.sum(labels[:, list(function)])
        total += labels.shape[0]
    expected_function_values /= total

    residuals = []
    ks = []
    worst_functions=[]
    for k in tqdm(range(1, 10000, 100)):
        runs = []
        worst_function = []
        for _ in range(5):
            # random_indices = torch.randperm(len(dataset))[:k].tolist()
            # random_indices = torch.random.randint(high=len(dataset), size=(k)).tolist()
            sampler =  torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=k)
            # subset = torch.utils.data.Subset(dataset, random_indices)
            subsetloader = torch.utils.data.DataLoader(dataset, sampler = sampler, batch_size=k)
            _, random_labels = next(iter(subsetloader))

            retrieved_function_value = torch.zeros(len(functions))
            for index, function in enumerate(functions):
                retrieved_function_value[index] = torch.linalg.norm(random_labels[:, list(function)].float())
            retrieved_function_value /= k

            runs.append(torch.max(torch.abs(retrieved_function_value-expected_function_values)).item())
            worst_function.append(torch.argmax(torch.abs(retrieved_function_value-expected_function_values)).item())
        worst_functions.extend(worst_function)
        residuals.extend(runs)
        ks.extend([k]*5)

    worst_function_indices = functions[max(set(worst_functions), key=worst_functions.count)]
    print("Worst function: ", worst_function_indices)
    print("Worst attributes: ", [attributes[i] for i in worst_function_indices])
    
    plotdf = pd.DataFrame({"k": ks, "residual": residuals})

    sns.lineplot(data=plotdf, x="k", y="residual")
    plt.xlabel("Retrieval size, k")
    plt.ylabel("Norm Difference")
    plt.title("Worst case norm difference for tree of depth " + str(depth))
    plt.savefig("./results/residuals_depth_{}.png".format(depth))

if __name__ == "__main__":
    main()