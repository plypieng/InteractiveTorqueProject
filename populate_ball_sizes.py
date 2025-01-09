# populate_ball_sizes.py
from app.database.session import SessionLocal
from app.database.models import BallSize
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, filename="populate_ball_sizes.log", format="%(asctime)s %(levelname)s:%(message)s"
)

def populate_ball_sizes():
    session = SessionLocal()
    try:
        # Define ball sizes and their torque ranges
        ball_sizes = [
            {"size": "Small", "torque_min": 0.0, "torque_max": 2.0},
            {"size": "Medium", "torque_min": 2.0, "torque_max": 8.0},
            {"size": "Large", "torque_min": 8.0, "torque_max": 12.0},
            # Add more ball sizes as needed
        ]

        for bs in ball_sizes:
            existing = session.query(BallSize).filter(BallSize.size == bs["size"]).first()
            if not existing:
                ball_size = BallSize(
                    size=bs["size"],
                    torque_min=bs["torque_min"],
                    torque_max=bs["torque_max"]
                )
                session.add(ball_size)
                logging.info(f"Added ball size: {bs['size']}")
            else:
                logging.info(f"Ball size already exists: {bs['size']}")

        session.commit()
        logging.info("Ball sizes populated successfully.")
    except Exception as e:
        logging.error(f"Error populating ball sizes: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    populate_ball_sizes()
