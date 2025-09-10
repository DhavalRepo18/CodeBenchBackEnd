

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_directory", type=str, default="./localtemp/trajectory/")
    parser.add_argument("--backstage_directory", type=str, default=".")

    args = parser.parse_args()
    traj_directory = args.traj_directory
    ans = get_failure_modes(traj_directory)
    
