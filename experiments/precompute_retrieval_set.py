from representational_retrieval import *
import torch
import numpy as np
from tqdm.auto import tqdm
import argparse
import clip
import debias_clip as dclip
import faiss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default="celeba", type=str)
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-query', default="queries.txt", type=str)
    parser.add_argument('-model', default="clip", type=str)
    args = parser.parse_args()
    print(args)

    if args.model=="clip":
        embedding_model = "clip"
        model, preprocess = clip.load("ViT-B/32", device=args.device)
    elif args.model == "debiasclip":
        embedding_model = "debiasclip"
        model, preprocess = dclip.load("ViT-B/16-gender", device =args.device) # DebiasClip for gender, the only publicly available model
        model.to(args.device)

    if args.dataset == "fairface":
        dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "occupations":
        dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "utkface":
        dataset = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "celeba":
        dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess, embedding_model=embedding_model)
    else:
        print("Dataset not supported!")
        exit()


    labels, embeds = dataset.labels, dataset.embeds
    embeds = torch.nn.functional.normalize(embeds, dim=0)
    print(labels.shape)
    print(embeds.shape)

    index = faiss.IndexFlatL2(512)   # build the index
    # res = faiss.StandardGpuResources()  # use a single GPU
    # build a flat (CPU) index
    # index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    # index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index.add(embeds.cpu())                  # add vectors to the index

    with open(args.query, 'r') as f:
        queries = f.readlines()
    q = None
    for query in queries:
        q_org = query.strip()
        qtext = "A photo of "+ q_org
        print(qtext)

        q_token = clip.tokenize(qtext).to(args.device)

        # ensure on the same device
        # q_token = q_token.to(args.device)

        with torch.no_grad():
            q_emb = model.encode_text(q_token).cpu().to(torch.float32)
        q_emb = q_emb/torch.nn.functional.normalize(q_emb, dim=0)
        
        if q is None:
            q = q_emb
        else:
            q = torch.cat((q, q_emb), dim=0)

    k = 10000                        

    D, I = index.search(q, k)     # actual search

    
    for i, q in enumerate(queries):
        query = q.strip()
        querytag = query.split(" ")[-1]

        images = []
        retrievals = []
        for index in I[i]:
            path = dataset.img_paths[index]
            image_id = path.split(".")[0]
            images.append(image_id)
            retrievals.append(embeds[index].cpu().numpy())
        
        if not os.path.isdir(os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets",args.dataset, embedding_model,querytag)):
            os.mkdir(os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets",args.dataset, embedding_model,querytag))
        
        save_path = os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets",args.dataset, embedding_model,querytag)
        # retrievals = torch.tensor(retrievals)
        # torch.save(retrievals, os.path.join(save_path, "100kembeds.pt"))
        retrievals = np.array(retrievals)
        np.save(os.path.join(save_path, "embeds.npy"), retrievals)
        with open(os.path.join(save_path, "images.txt"), "w") as f:
            for idd in images:
                f.write(idd + "\n")

if __name__ == "__main__":
    main()