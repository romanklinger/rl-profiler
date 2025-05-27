def calculate_reward(selected_tweets, y_true, llm=None):
    y_pred = "None"
    if llm:
        reward = 0
        selected_tweets_num = len(selected_tweets)

        if len(selected_tweets) == 0:
            reward = -2
        else:
            y_pred = llm.evaluate_author(selected_tweets, y_true)
            if y_pred == y_true:
                reward = +1
            else:
                reward = -1
        reward = reward - (0.05 * selected_tweets_num)

    # Early development without LLM
    else:
        selected_tweets_num = len(selected_tweets)
        selected_tweets = ' '.join(selected_tweets)
        if "sample tweet" in selected_tweets.lower():
            reward = 0
        else:
            reward = +1
        if selected_tweets_num == 0:
            reward = -1
        reward = reward - (0.05 * selected_tweets_num)

    return reward, y_pred
