import argparse
import torch
import json
import predict_utils

def get_command_line_args():
    parser = argparse.ArgumentParser()
    #-----Required Arguments----------
    parser.add_argument('input', type=str,
                        help='Image file')
    
    parser.add_argument('checkpoint', type=str,
                        help='Saved model checkpoint')

    #-----Optional Arguments----------
        
    parser.add_argument('--top_k', type=int,
                        help='Return the top K most likely classes')
    parser.set_defaults(top_k=1)
    
    parser.add_argument('--category_names', type=str,
                        help='File of category names')

    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Use GPU')
    parser.set_defaults(gpu=False)

    return parser.parse_args()


def main():
    # Get input arguments
    args = get_command_line_args()
    use_gpu = torch.cuda.is_available() and args.gpu
    print("Input file: {}".format(args.input))
    print("Checkpoint file: {}".format(args.checkpoint))
    if args.top_k:
        print("Returning {} most likely classes".format(args.top_k))
    if args.category_names:
        print("Category names file: {}".format(args.category_names))
    if use_gpu:
        print("Using GPU.")
    else:
        print("Using CPU.")
    
    # Load the checkpoint
    model = predict_utils.load_checkpoint(args.checkpoint)
    print("Checkpoint loaded.")
    
    # Move tensors to GPU
    if use_gpu:
        model.cuda()
    
    # Load categories file
    if args.category_names:
        with open(args.category_names, 'r') as f:
            categories = json.load(f)
            print("Category names loaded")
    
    results_to_show = args.top_k if args.top_k else 1
    
    # Predict
    print("Processing image")
    probabilities, classes = predict_utils.predict(args.input, model, use_gpu, results_to_show, args.top_k)
    
    # Show the results
    # Print results
    if results_to_show > 1:
        print("Top {} Classes for '{}':".format(len(classes), args.input))

        if args.category_names:
            print("{:<30} {}".format("Flower", "Probability"))
            print("------------------------------------------")
        else:
            print("{:<10} {}".format("Class", "Probability"))
            print("----------------------")

        for i in range(0, len(classes)):
            if args.category_names:
                print("{:<30} {:.2f}".format(categories[classes[i]], probabilities[i]))
            else:
                print("{:<10} {:.2f}".format(classes[i], probabilities[i]))
    else:
        print("The most likely class is '{}': probability: {:.2f}" \
              .format(categories[classes[0]] if args.category_names else classes[0], probabilities[0]))
        
    
if __name__ == "__main__":
    main()