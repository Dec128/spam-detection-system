Submission Requirements for Program Code & Video
You are required to use Python, Java, or VB.Net for your application development. If your application requires the use of other programming languages, please consult with your teacher. 

COMPONENT A: PROJECT CODE 
Use an IDE to develop the program code for your module (including comments). Include comments or acknowledgments for any part of the submitted program code that has been written by someone other than you and/or your collaborative partner(s).
In your program, you must include student-developed program code that contains the following: 
□ Instructions for input from one of the following:
◆ the user (including user actions that trigger events)
◆ a device
◆ an online data stream
◆ a file
□Use of at least one list (or other collection type) to represent a collection of data that is stored and used to manage program complexity and help fulfill the program’s purpose




□ At least one subroutine that contributes to the program’s intended purpose, where you have 
    defined: 
◆ the subroutine’s name 
◆ the return type (if necessary) 
◆ one or more parameters
 □ A program that includes sequencing, selection, and iteration that is in the body of the 
     selected module
 □ Calls to your student-developed procedure 
 □ Output (tactile, audible, visual, or textual) with appropriate messages based on input and  
    program functionality
COMPONENT B: VIDEO DEMONSTRATION 
Submit one video file that demonstrates the running of your program code as described below. 
Your video must demonstrate your program running, including: 
□ Input to your program
□ Integration with another module developed by a group member
□ Output produced by your program
□ A working module that meets the requirements of the project.
Your video may NOT contain:
□ Any distinguishing information about yourself
□ Voice narration (though text captions are encouraged)
Your video must be:
□ Either .webm, .mp4, .wmv, .avi, or .mov format
□ No more than 1 minute in length
□ No more than 30MB in file size


# review_module.py

# Data storage
reviews = []
moderated_reviews = []

# Helper function to find an order by ID
def find_order_by_id(order_id, orders):
    for order in orders:
        if order['order_id'] == order_id:
            return order
    return None

# Function to submit a review
def submit_review(order_id, review_text, rating, reviews_list):
    review_id = len(reviews_list) + 1
    review = {
        'review_id': review_id,
        'order_id': order_id,
        'review_text': review_text,
        'rating': rating,
        'approved': False
    }
    reviews_list.append(review)
    return review

# Function to rate a restaurant
def rate_restaurant(order_id, rating, reviews_list):
    for review in reviews_list:
        if review['order_id'] == order_id:
            review['rating'] = rating
            return review
    return None

# Function for admin to moderate reviews
def moderate_review(review_id, reviews_list, moderated_list, approve):
    for review in reviews_list:
        if review['review_id'] == review_id:
            review['approved'] = approve
            moderated_list.append(review)
            reviews_list.remove(review)
            return review
    return None

# Function to get reviews for a restaurant
def get_reviews_for_restaurant(restaurant_id, reviews_list, orders_list):
    restaurant_reviews = []
    for review in reviews_list:
        order = find_order_by_id(review['order_id'], orders_list)
        if order and order['restaurant_id'] == restaurant_id and review['approved']:
            restaurant_reviews.append(review)
    return restaurant_reviews

# Function to display reviews
def display_reviews(restaurant_reviews):
    for review in restaurant_reviews:
        print(f"Order ID: {review['order_id']}, Review: {review['review_text']}, Rating: {review['rating']}")

# Sample usage
def main():
    orders = [
        {'order_id': 1, 'user_id': 1, 'restaurant_id': 1, 'items': ['Margherita']},
        {'order_id': 2, 'user_id': 2, 'restaurant_id': 2, 'items': ['Cheeseburger']}
    ]

    user1 = {'user_id': 1, 'username': 'john_doe', 'password': 'password123'}
    user2 = {'user_id': 2, 'username': 'jane_smith', 'password': 'securepass'}

    restaurant1 = {'restaurant_id': 1, 'name': 'Pizza Place', 'menu': [('Margherita', 8), ('Pepperoni', 10)]}
    restaurant2 = {'restaurant_id': 2, 'name': 'Burger Joint', 'menu': [('Cheeseburger', 5), ('Veggie Burger', 6)]}

    # Submit reviews
    review1 = submit_review(1, 'Great pizza!', 5, reviews)
    review2 = submit_review(2, 'Delicious burger!', 4, reviews)

    # Admin moderates reviews
    moderated_review1 = moderate_review(review1['review_id'], reviews, moderated_reviews, True)
    moderated_review2 = moderate_review(review2['review_id'], reviews, moderated_reviews, False)

    # Display approved reviews for Pizza Place
    restaurant1_reviews = get_reviews_for_restaurant(1, moderated_reviews, orders)
    print("\nApproved Reviews for Pizza Place:")
    display_reviews(restaurant1_reviews)

if __name__ == "__main__":
    main()
