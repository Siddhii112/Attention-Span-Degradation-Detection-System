def get_recommendation(category):
    if category == "Low":
        return (
            "🔴 Low Attention:\n"
            "1. Minimize distractions (avoid mobile phones, social media)\n"
            "2. Use Pomodoro technique (25 min study + 5 min break)\n"
            "3. Study in a quiet and comfortable environment\n"
            "4. Break large tasks into smaller, manageable parts\n"
            "5. Maintain a consistent daily study schedule"
        )

    elif category == "Medium":
        return (
            "🟠 Medium Attention:\n"
            "1. Follow a structured study plan with clear goals\n"
            "2. Take short breaks to maintain concentration\n"
            "3. Engage in active learning (notes, quizzes, discussions)\n"
            "4. Increase interaction with learning materials (videos, practice)\n"
            "5. Track your progress regularly"
        )

    else:
        return (
            "🟢 High Attention:\n"
            "1. Maintain your current study habits consistently\n"
            "2. Challenge yourself with advanced topics\n"
            "3. Help peers to reinforce your understanding\n"
            "4. Explore additional learning resources\n"
            "5. Stay consistent and avoid burnout with balanced breaks"
        )