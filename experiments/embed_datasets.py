import debias_clip as dclip
import clip
import argparse
from representational_retrieval import *
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default="clip", type=str)
    parser.add_argument('-dataset', default="celeba", type=str)
    parser.add_argument('-device', default="cuda", type=str)
    args = parser.parse_args()


    model_type = args.model
    if args.model == "clip":
        usingclip = True
        model, preprocess = clip.load("ViT-B/32", device=args.device)
    elif args.model == "debiasclip":
        usingclip = False
        model, preprocess = dclip.load("ViT-B/16-gender", device =args.device) # DebiasClip for gender, the only publicly available model
    else:
        usingclip = True
        model, preprocess = clip.load("ViT-B/32", device=args.device)
        model_type = "clip"

    if args.dataset == "fairface":
        # dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=False, transform=preprocess, return_path=True)
        print("Alex already embedded, talk to him")
        exit()
    elif args.dataset == "occupations":
        dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, return_path=True)
        # print("Alex already embedded, talk to him")
        # exit()
    elif args.dataset == "utkface":
        dataset = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, return_path=True)
    elif args.dataset == "celeba":
        print("Alex already embedded, talk to him")
        exit()
        # dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=False, transform=preprocess, return_path=True)
    else:
        print("Dataset not supported!")
        exit()

    if not os.path.isdir(os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets/", args.dataset, model_type)):
        os.mkdir(os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets/", args.dataset, model_type))
    clipdir = os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets/", args.dataset, model_type)

    a = False
    with torch.no_grad():
        for images, _, path in tqdm(DataLoader(dataset, batch_size=512)):
            features = model.encode_image(images.to(args.device))
            for i in range(features.shape[0]):
                path_i = path[i]
                image_id = path_i.split(".")[0]
                if not os.path.isdir(os.path.dirname(os.path.join(clipdir, image_id+".pt"))):
                    os.mkdir(os.path.dirname(os.path.join(clipdir, image_id+".pt")))
                if a == False:
                    print(image_id)
                    print(os.path.dirname(os.path.join(clipdir, image_id+".pt")))
                    a = True
                torch.save(features[i].squeeze(), os.path.join(clipdir, image_id+".pt"))

if __name__ == "__main__":
    main()