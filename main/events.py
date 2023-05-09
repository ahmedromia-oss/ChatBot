from flask import session
from flask_socketio import emit, join_room, leave_room
from .. import socketio
from ..MainAI.chatbot import Bot

@socketio.on('joined', namespace='/chat')
def joined(message):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    name = session.get('name')
    room = session.get('room')
    join_room(room)
    emit('status', {'msg': session.get('name') + '! nice to have you here'}, room=room)


@socketio.on('text', namespace='/chat')
def text(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    room = session.get('room')
    emit('message', {'msg': message['msg']}, room=room)

@socketio.on('botResponse', namespace='/chat')
def BotResponse(message):
    room = session.get('room')
    emit('messageBot', {'msg': process(message['msg'])}, room=room)
    


def process(message):
    return Bot(message)




@socketio.on('left', namespace='/chat')
def left(message):
    """Sent by clients when they leave a room.
    A status message is broadcast to all people in the room."""
    room = session.get('room')
    leave_room(room)
    emit('status', {'msg': session.get('name') + ' has left the room.'}, room=room)

