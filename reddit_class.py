# Richard Joerger

import praw
import sys
import csv
import json
from time import sleep

USER_FILE = "users.csv"


def get_info(which):
    """
    This opens the file which contains user credentials. 
    """
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
    """
    The main function. It handles actually getting the data.
    """
    if len(sys.argv) != 4:
        print("python2 reddit_class.py <user> <subreddit> <# of data_points>")
        exit()

    # Reading command line args as well as loading up the information from the
    # user accounts into PRAW
    subreddit = sys.argv[2]
    data_points = int(sys.argv[3])
    vals = get_info(int(sys.argv[1]))
    reddit = praw.Reddit(client_id=vals['client'],
                         client_secret=vals['secret'],
                         password=vals['password'],
                         user_agent=vals['user_agent'],
                         username=vals['username'])

    # Just going to loop indefinitely and let the inside handle exiting.
    while True:

        # Dictionary of comments where the key is the post id (which is unique)
        # and the value is the dictionary of the post.
        data = {}
        try:
            # "iterating" on the comment stream.
            for comment in \
                            reddit.subreddit(subreddit).stream.comments():

                # Making another request for the comment information.
                in_data = vars(comment)

                # Now that we have the data for the comment lets just extract
                # what we need and store it. For us this means the body, the up
                # vtes, the score, the number of comments associated, how 
                # controverisal the post is, how many down votes, if the 
                # comment comes from the OP, who the author of the comment was
                # and the actual post id. Then we save that all in a dictionary
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
                # We check if the comment is not already in the dictionary
                if comm_id not in data:
                    # If it's not we save it and subtract one from our total
                    # points and print an info message. 
                    data[comm_id] = to_save
                    data_points = data_points - 1
                    print("%s: %d" % (subreddit, data_points))
                    # if we've gotten all of the points we need...
                    if data_points == 0:
                        # we let the user know and throw an exception. 
                        print("Got all them points")
                        raise Exception()
                    # The comment was already seen so we'll just choose to do
                    # nothing. 
        except Exception as e:
            # We're catching an exception, we don't know from where so lets
            # print the exception message
            print(e)

            # lets also check how many data points we have
            if data_points == 0:
                # We've gotten our exception from the parser, lets exit the loop
                # and therefore the program. 
                break
            #In this case, we didn't leave, we're going to wait. 
            print("starting 10 minute sleep")
            sleep(600)
            print("done sleeping")
    with open('data_%s.txt' % subreddit, 'w') as outfile:
        # now that we have all our points, lets dump the python dictionary into
        # a json formatted one and into a file. 
        outfile.write(json.dumps(data))
            

if __name__ == '__main__':
    main()
