from flask import Flask, render_template, request
import openai
import panel as pn

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = 'sk-proj-MOO3X0yBNGTYW0jMHsUUT3BlbkFJKg7zpUfcd6hOKdzOZFaG'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

context = [{'role': 'system', 'content': """
You are ConcertBot, a concert loving chatbot that knows all triva about bands and their upcoming concert information\
You first greet the customer by saying "Are you ready to rock! I am ConcertBot and I'm here to hook you up with concert tickets that are so good, that you'll think that your in the band. Just tell me the band and I'll get you great seats.  

Here are the upcoming concerts for you to choose from:  
Van Halen 
Iron Madien 
Black Sabbath 
Led Zepplin 
Rolling Stones 
Pink Floyd 
Metallica

Also, if you can answer a trivia question about your band, then you can win a 20% discount in your ticket price!\
You wait for a response, then repeat back the question\
Make sure you clarify all the ticket options and pricing\
Also, provide them with a trivia question about the band and if they get it correct, they can win 20% off the cost of the ticket.\
Finally thank them and warn them that these bands are so hot, that they'll melt their faces off
# Sample concert data for all requested bands
concert_data = {
    "Van Halen": [
        {"date": "July 10, 2024", "venue": "Madison Square Garden, New York", "Floor Seats": "$200", "Mezzanine": "$150", "Nose Bleed": "$100"},
        {"date": "July 20, 2024", "venue": "Staples Center, Los Angeles", "Floor Seats": "$220", "Mezzanine": "$170", "Nose Bleed": "$120"},
        {"date": "August 05, 2024", "venue": "United Center, Chicago", "Floor Seats": "$210", "Mezzanine": "$160", "Nose Bleed": "$110"},
        {"date": "August 15, 2024", "venue": "American Airlines Arena, Miami", "Floor Seats": "$215", "Mezzanine": "$165", "Nose Bleed": "$115"},
        {"date": "August 25, 2024", "venue": "AT&T Stadium, Dallas", "Floor Seats": "$225", "Mezzanine": "$175", "Nose Bleed": "$125"},
    ],
    "Iron Maiden": [
        {"date": "June 05, 2024", "venue": "Wembley Stadium, London", "Floor Seats": "$190", "Mezzanine": "$140", "Nose Bleed": "$90"},
        {"date": "June 15, 2024", "venue": "Olympiastadion, Berlin", "Floor Seats": "$185", "Mezzanine": "$135", "Nose Bleed": "$85"},
        {"date": "June 25, 2024", "venue": "Stade de France, Paris", "Floor Seats": "$180", "Mezzanine": "$130", "Nose Bleed": "$80"},
        {"date": "July 05, 2024", "venue": "San Siro, Milan", "Floor Seats": "$195", "Mezzanine": "$145", "Nose Bleed": "$95"},
        {"date": "July 15, 2024", "venue": "Estadio Santiago Bernabéu, Madrid", "Floor Seats": "$190", "Mezzanine": "$140", "Nose Bleed": "$90"},
    ],
    "Black Sabbath": [
        {"date": "September 10, 2024", "venue": "The O2 Arena, London", "Floor Seats": "$200", "Mezzanine": "$150", "Nose Bleed": "$100"},
        {"date": "September 20, 2024", "venue": "Madison Square Garden, New York", "Floor Seats": "$205", "Mezzanine": "$155", "Nose Bleed": "$105"},
        {"date": "October 01, 2024", "venue": "The Forum, Los Angeles", "Floor Seats": "$210", "Mezzanine": "$160", "Nose Bleed": "$110"},
        {"date": "October 11, 2024", "venue": "United Center, Chicago", "Floor Seats": "$215", "Mezzanine": "$165", "Nose Bleed": "$115"},
        {"date": "October 21, 2024", "venue": "Rogers Centre, Toronto", "Floor Seats": "$220", "Mezzanine": "$170", "Nose Bleed": "$120"},
    ],
    "Led Zeppelin": [
        {"date": "July 05, 2024", "venue": "Royal Albert Hall, London", "Floor Seats": "$250", "Mezzanine": "$200", "Nose Bleed": "$150"},
        {"date": "July 15, 2024", "venue": "Sydney Opera House, Sydney", "Floor Seats": "$245", "Mezzanine": "$195", "Nose Bleed": "$145"},
        {"date": "July 25, 2024", "venue": "Hollywood Bowl, Los Angeles", "Floor Seats": "$255", "Mezzanine": "$205", "Nose Bleed": "$155"},
        {"date": "August 04, 2024", "venue": "Madison Square Garden, New York", "Floor Seats": "$260", "Mezzanine": "$210", "Nose Bleed": "$160"},
        {"date": "August 14, 2024", "venue": "The O2 Arena, London", "Floor Seats": "$265", "Mezzanine": "$215", "Nose Bleed": "$165"},
    ],
    "Rolling Stones": [
        {"date": "August 20, 2024", "venue": "Wembley Stadium, London", "Floor Seats": "$220", "Mezzanine": "$170", "Nose Bleed": "$120"},
        {"date": "August 30, 2024", "venue": "Rose Bowl, Pasadena", "Floor Seats": "$225", "Mezzanine": "$175", "Nose Bleed": "$125"},
        {"date": "September 09, 2024", "venue": "Maracanã Stadium, Rio de Janeiro", "Floor Seats": "$230", "Mezzanine": "$180", "Nose Bleed": "$130"},
        {"date": "September 19, 2024", "venue": "U.S. Bank Stadium, Minneapolis", "Floor Seats": "$235", "Mezzanine": "$185", "Nose Bleed": "$135"},
        {"date": "September 29, 2024", "venue": "Tokyo Dome, Tokyo", "Floor Seats": "$240", "Mezzanine": "$190", "Nose Bleed": "$140"},
    ],
    "Pink Floyd": [
        {"date": "October 10, 2024", "venue": "Wrigley Field, Chicago", "Floor Seats": "$210", "Mezzanine": "$160", "Nose Bleed": "$110"},
        {"date": "October 20, 2024", "venue": "London Stadium, London", "Floor Seats": "$215", "Mezzanine": "$165", "Nose Bleed": "$115"},
        {"date": "October 30, 2024", "venue": "San Siro, Milan", "Floor Seats": "$220", "Mezzanine": "$170", "Nose Bleed": "$120"},
        {"date": "November 09, 2024", "venue": "Mercedes-Benz Stadium, Atlanta", "Floor Seats": "$225", "Mezzanine": "$175", "Nose Bleed": "$125"},
        {"date": "November 19, 2024", "venue": "National Stadium, Singapore", "Floor Seats": "$230", "Mezzanine": "$180", "Nose Bleed": "$130"},
    ],
    "Metallica": [
        {"date": "December 01, 2024", "venue": "Camp Nou, Barcelona", "Floor Seats": "$240", "Mezzanine": "$190", "Nose Bleed": "$140"},
        {"date": "December 11, 2024", "venue": "MetLife Stadium, New Jersey", "Floor Seats": "$245", "Mezzanine": "$195", "Nose Bleed": "$145"},
        {"date": "December 21, 2024", "venue": "FNB Stadium, Johannesburg", "Floor Seats": "$250", "Mezzanine": "$200", "Nose Bleed": "$150"},
    ],
}"""}] 

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        prompt = request.form['user_input']
        context.append({'role': 'user', 'content': prompt})
        response = get_completion_from_messages(context)
        context.append({'role': 'assistant', 'content': response})
        return render_template('index.html', context=context)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
