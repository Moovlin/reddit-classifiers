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
            for comment in reddit.subreddit(subreddit).stream.comments():
                in_data = vars(comment)
                title = in_data.link_title
                body = in_data.body
                to_save = {'title': title, 'body': body}
                data[str(data_points)] = to_save
                if data_points == 0:
                    raise Exception()
                data_points = data_points - 1
        except (Exception):
            if data_points == 0:
                break
            print("starting 15 minute sleep")
            sleep(900)
            print("done sleeping")
    with open('data_%s.txt' % subreddit, 'w') as outfile:
        outfile.write(str(data))
            

if __name__ == '__main__':
    main()
