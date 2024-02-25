# -*- coding: cp1252 -*-
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.core.window import Window

class MyLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MyLayout, self).__init__(**kwargs)
        self.text_input1 = TextInput()
        self.text_input2 = TextInput()
        self.add_widget(self.text_input1)
        self.add_widget(self.text_input2)
        Window.bind(on_key_down=self.on_keyboard_down)

    def on_keyboard_down(self, window, key, scancode, codepoint, modifier):
        if key == 9:  # Tab key
            if self.text_input1.focus:
                self.text_input2.focus = True
                self.text_input1.focus = False
            else:
                self.text_input1.focus = True
                self.text_input2.focus = False

import re

def remove_punctuation_and_digits(text):
    # Remove punctuation and digits, keep only letters
    text = re.sub(r'[^a-zA-Z]', '', text)

    return text

# 示例用法
text = "Hello, 123 world!"
processed_text = remove_punctuation_and_digits(text)
print(processed_text)



# class MyApp(App):
#     def build(self):
#         return MyLayout()
#
# if __name__ == '__main__':
#     MyApp().run()
