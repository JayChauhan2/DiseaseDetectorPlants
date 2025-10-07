from flask_wtf import FlaskForm
from flask_wtf.file import FileField

class uploadForm(FlaskForm):
    userImg = FileField('') #img upload
    