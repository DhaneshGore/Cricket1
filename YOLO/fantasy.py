# ==== Fantasy Score Overlay Functions ====

# Initial fantasy scores (can be loaded dynamically if needed)
fantasy_scores = {
    "player1": 50,
    "player2": 30,
    "player3": 70,
}

def get_current_scores():
    return fantasy_scores

def get_hypothetical_scores(event, outcome, player="player1"):
    scores = fantasy_scores.copy()
    if event == "catch" and outcome == "dropped":
        scores[player] += 10  # Example logic
    return scores

def draw_fantasy_scores_on_frame(frame, scores, title="Fantasy Scores"):
    """
    Draws fantasy scores as overlay text on the given frame.
    """
    y = 40
    cv2.putText(frame, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for player, score in scores.items():
        y += 30
        cv2.putText(frame, f"{player}: {score}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame
