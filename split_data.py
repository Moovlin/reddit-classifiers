# Richard Joerger
import json

# We know what our sources are so we won't implement them nicely.
sources = ["data_askreddit.txt", "data_personalfinance.txt",
"data_showerthoughts.txt"]


# For source in our source list....
for source in sources:
    # Create a temp dictionary to load the json dictionary into
    json_data = {}
    with open(source) as f:
        # Loading the dictionary.
        temp = json.loads(f.read())

    # Now we have to see how many items are in dictionary and then do some math.
    total_items = len(temp.items())

    num_train = int(total_items * .5) + 1   # Training set size
    num_dev = int(total_items * .25)    # Development size
    num_test = int(total_items * .25)   # Testing set size

    spot = 0    # Keeping track of how many spots we've filled
    train_temp = {}     # The temp dictionary for the training set.
    total_keys = temp.keys()    # Getting the keys from the total set such that
                                # we can iterate over them.

    # Just getting the correct number of keys
    for i in range(spot, num_train):
        train_temp[total_keys[i]] = temp[total_keys[i]]

    # We've gotten all of the keys for the training set, now we need to get the
    # keys for the development set.
    spot = num_train - 1
    dev_temp = {}
    for i in range(spot, spot + num_dev):
        print(i)
        dev_temp[total_keys[i]] = temp[total_keys[i]]

    spot = spot + num_dev

    # We're repeating the process for the testing set.
    test_temp = {}
    for i in range(spot, spot + num_test):
        test_temp[total_keys[i]] = temp[total_keys[i]]

    # Just writing to the files with the correct name. 
    print(len(train_temp), len(dev_temp), len(test_temp))
    with open('train_%s' % source, 'w') as f:
        f.write(json.dumps(train_temp))

    with open('dev_%s' % source, 'w') as f:
        f.write(json.dumps(dev_temp))

    with open('test_%s' % source, 'w') as f:
        f.write(json.dumps(test_temp))
