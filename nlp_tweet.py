from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey="wJnqypSNVN0rtd00AL9dP0i6f"
csecret="KdTcj1LbaZwaVxL9sabMWQe0QlaxQLvrHYXwNJqffH4MZY9BKj"
atoken="1079256461983674368-vQGeoOMEjemdHV69aUwQ1cRkf399YT"
asecret="6y89dDAAPIxkLNQ1uJAgjcIuE3gY1OiObX2BHINgF8blL"

class listener(StreamListener):

    def on_data(self, data):
        try:
            all_data = json.loads(data)

            tweet = all_data["text"]
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence)

            if confidence*100 >= 80:
                output = open("twitter-out.txt", "a")
                output.write(sentiment_value)
                output.write("\n")
                output.close()

            return True
        except:
            return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])