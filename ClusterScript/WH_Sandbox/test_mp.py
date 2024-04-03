import numpy as np
import multiprocessing as mp

def my_function(arg1, arg2, common_arg):
    # Your function implementation here
    result = arg1 + arg2 + common_arg
    return result, arg1

def main():
    # Define your list of arguments
    arg_list = [(1, 2), (3, 4), (5, 6)]  # Each tuple represents arguments for one function call

    # Define common arguments
    common_arg = 10

    # Create a Pool object within a context manager
    num_processes = mp.cpu_count()  # You can change the number of processes as needed
    with mp.Pool(processes=num_processes) as pool:

        # Call my_function with the arguments in arg_list using Pool.map
        results = pool.starmap(my_function, [(args[0], args[1], common_arg) for args in arg_list])

    # Print the results
    print(results)

    #Checking ignoring the return values
    _,_ = my_function(1,2,3)
    print("passed")


    #checking savelist condition 
    lambsavelist = np.array([0.1,0.05,0.01,0.005,0.001])
    lamblooplist = np.arange(1,0.001 - 1e-10,-0.001)
    lamb = lamblooplist[900]
    print(f'lamb = {lamb}')
    print(np.isclose(lambsavelist, lamb))
    print(lambsavelist[np.isclose(lambsavelist, lamb)][0])

if __name__ == "__main__":
    main()
