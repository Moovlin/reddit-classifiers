import json

sources = ["data_askreddit.txt", "data_personalfinance.txt",
"data_showerthoughts.txt"]



for source in sources:
    json_data = {}
    with open(source) as f:
        temp = json.loads(f.read())

    total_items = len(temp.items())
    num_train = int(total_items * .5) + 1
    num_dev = int(total_items * .25)
    num_test = int(total_items * .25)

    spot = 0
    train_temp = {}
    total_keys = temp.keys()
    for i in range(spot, num_train):
        train_temp[total_keys[i]] = temp[total_keys[i]]

    spot = num_train - 1
    dev_temp = {}
    for i in range(spot, spot + num_dev):
        print(i)
        dev_temp[total_keys[i]] = temp[total_keys[i]]

    spot = spot + num_dev

    test_temp = {}
    for i in range(spot, spot + num_test):
        test_temp[total_keys[i]] = temp[total_keys[i]]

    print(len(train_temp), len(dev_temp), len(test_temp))
    with open('train_%s' % source, 'w') as f:
        f.write(json.dumps(train_temp))

    with open('dev_%s' % source, 'w') as f:
        f.write(json.dumps(dev_temp))

    with open('test_%s' % source, 'w') as f:
        f.write(json.dumps(test_temp))
