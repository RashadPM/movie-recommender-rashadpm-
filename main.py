from src.data_preprocessing import load_and_merge, clean_data, save_processed
from src.train_model import train
from src.recommend import recommend


def main():

    print("🚀 Starting full pipeline...")

    # -------- Preprocessing --------
    print("📥 Loading raw datasets...")
    movies = load_and_merge()

    print("🧹 Cleaning data...")
    movies = clean_data(movies)

    print("💾 Saving processed dataset...")
    save_processed(movies)

    # -------- Model Training --------
    print("🎯 Starting model training...")
    train()

    # -------- Testing Recommendation --------
    print("\n🎬 Testing recommendation system...")
    results = recommend("Avatar")

    for movie in results:
        print("👉", movie)

    print("\n🎉 FULL PIPELINE COMPLETED!")


if __name__ == "__main__":
    main()