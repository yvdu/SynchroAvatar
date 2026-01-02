import sys
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QFileDialog, QLineEdit, QComboBox, QLabel
)
from main_backend2 import run_pipeline

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Run Backend Script')

        layout = QVBoxLayout()

        # ---- Image selection ----
        self.image_label = QLabel('Select Image:', self)
        self.image_button = QPushButton('Browse Image', self)
        self.image_button.clicked.connect(self.browse_image)
        self.image_path = ''
        layout.addWidget(self.image_label)
        layout.addWidget(self.image_button)

        # ---- Audio selection ----
        self.audio_label = QLabel('Select Audio:', self)
        self.audio_button = QPushButton('Browse Audio', self)
        self.audio_button.clicked.connect(self.browse_audio)
        self.audio_path = ''
        layout.addWidget(self.audio_label)
        layout.addWidget(self.audio_button)

        # ---- Background selection ----
        self.bg_label = QLabel('Select Background Image:', self)
        self.bg_button = QPushButton('Browse Background', self)
        self.bg_button.clicked.connect(self.browse_bg)
        self.bg_path = ''
        layout.addWidget(self.bg_label)
        layout.addWidget(self.bg_button)

        # ---- Position selection ----
        self.position_label = QLabel('Select Video Position:', self)
        self.position_combo = QComboBox(self)
        self.position_combo.addItem('左上', 'left_top')
        self.position_combo.addItem('右上', 'right_top')
        self.position_combo.addItem('左下', 'left_bottom')
        self.position_combo.addItem('右下', 'right_bottom')
        self.position_combo.addItem('中间', 'center')
        layout.addWidget(self.position_label)
        layout.addWidget(self.position_combo)

        # ---- Text input ----
        self.text_label = QLabel('Enter Text:', self)
        self.text_input = QLineEdit(self)
        layout.addWidget(self.text_label)
        layout.addWidget(self.text_input)

        # ---- Language selection ----
        self.language_label = QLabel('Select Language:', self)
        self.language_combo = QComboBox(self)
        self.language_combo.addItem('Chinese', 'chinese')
        self.language_combo.addItem('English', 'english')
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_combo)

        # ---- Run button ----
        self.run_button = QPushButton('Run', self)
        self.run_button.clicked.connect(self.run_backend)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def browse_image(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 'Images (*.png *.jpg *.jpeg *.bmp *.gif)', options=options)
        if file:
            self.image_path = file
            self.image_label.setText(f'Selected Image: {file}')

    def browse_audio(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            self, 'Select Audio', '', 'Audio Files (*.mp3 *.wav *.flac)', options=options)
        if file:
            self.audio_path = file
            self.audio_label.setText(f'Selected Audio: {file}')

    def browse_bg(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            self, 'Select Background Image', '', 'Images (*.png *.jpg *.jpeg *.bmp *.gif)', options=options)
        if file:
            self.bg_path = file
            self.bg_label.setText(f'Selected Background: {file}')

    def run_backend(self):
        # Get the text, language, and position
        text = self.text_input.text()
        language = self.language_combo.currentData()
        position = self.position_combo.currentData()

        if not self.image_path or not self.audio_path or not text:
            self.image_label.setText("Error: Image, Audio, and Text are required.")
            return
        if not self.bg_path:
            self.bg_label.setText("Error: Background image is required.")
            return

        run_pipeline(audio_path=self.audio_path,
                     image_path=self.image_path,
                     out_text=text,
                     bg_img=self.bg_path,
                     position=position)

        # Prepare the command to run the backend script
        '''
        command = [
            'python3', 'main_backend2.py',
            '--audio', self.audio_path,
            '--image', self.image_path,
            '--bg_path', self.bg_path,
            '--position', position,
            '--text', text
        ]

        print("Running command:", " ".join(command))

        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                self.image_label.setText("✅ Backend script completed successfully.")
            else:
                self.image_label.setText(f"❌ Error: {result.stderr}")
        except Exception as e:
            self.image_label.setText(f"⚠️ Exception: {e}")
        '''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
