# Richard Joerger

import praw
import sys
import csv
import json
from time import sleep

USER_FILE = "users.csv"


def get_info(which):
    with open(USER_FILE) as csvfile:
        reader = csv.DictReader(csvfile)
        vals = []
        for row in reader:
            vals.append(row)
        if which == 1:
            # Load the dict for user 1
            return vals[0]
        else:
            # Load the dict for user 2
            return vals[1]


def main():
    if len(sys.argv) != 4:
        print("python2 reddit_class.py <user> <subreddit> <# of data_points>")
        exit()

    subreddit = sys.argv[2]
    data_points = int(sys.argv[3])
    vals = get_info(int(sys.argv[1]))
    reddit = praw.Reddit(client_id=vals['client'],
                         client_secret=vals['secret'],
                         password=vals['password'],
                         user_agent=vals['user_agent'],
                         username=vals['username'])

    while True:
        data = {}
        try:
            for comment in \
            reddit.subreddit(subreddit).stream.comments():
                in_data = vars(comment)
                title = in_data['link_title']
                body = in_data['body']
                ups = in_data['ups']
                score = in_data['score']
                num_comments = in_data['num_comments']
                contro = in_data['controversiality']
                downs = in_data['downs']
                submitter = in_data['is_submitter']
                author = in_data['author']
                comm_id = in_data['id']
                to_save = {"title": title, 
                           "body": body,
                           "author": author.fullname,
                           "ups": ups,
                           "downs": downs,
                           "submitter": submitter,
                           "score": score,
                           "num_comments": num_comments,
                           "contro": contro}
                if comm_id not in data:
                    data[comm_id] = to_save
                    data_points = data_points - 1
                    print("%s: %d" % (subreddit, data_points))
                    if data_points == 0:
                        print("Got all them points")
                        raise Exception()
        except Exception as e:
            print(e)
            if data_points == 0:
                break
            print("starting 10 minute sleep")
            sleep(600)
            print("done sleeping")
    with open('data_%s.txt' % subreddit, 'w') as outfile:
        outfile.write(json.dumps(data))
            

if __name__ == '__main__':
    main()
