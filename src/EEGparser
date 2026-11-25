from obci_readmanager.signal_processing.read_manager import ReadManager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import json 
import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

from abc import ABC


class Parsers(ABC):

    def __init__(self, metadata_path):

        with open(metadata_path, "r") as f:
            config = json.load(f)

    def read(self):
        """
        Returns a pd.DataFrame of shape (t_recording*fs, n_chanels), with time stamps as a indexes and channel names as columns
        """

        pass


class DummyParser(Parsers):


    def __init__(self, metadata_path):
        config = super().__init__(metadata_path)

        self.t_recording = 10
        self.fs = 512
        self.system = "1020"

        if self.system == "1020":

            ch_names = ["FP1","FP2","AFZ","F7","F3","FZ","F4","F8","REF","T3","C3","CZ","C4","T4","T5","P3","PZ","P4","T6","O1","O2"]

        elif self.system == "1010":

            ch_names = ["FC3","FC1","FCZ","FC2","FC4","A1","C5","C3","C1","CZ","C2","C4","C6","A2","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","P1","P2"]
            
        time = np.arange(0,self.t_recording, 1/self.fs)
        data_arr = np.random.normal(0,1, (self.t_recording*self.fs, len(ch_names)))
        self.data = pd.DataFrame.from_records(data_arr, index=time, columns=ch_names)


    def run(self):

        super().read()

        return self.data





class EEGparser(Parsers):



    POSITIONS_1020 = {
        # Linia czołowa (Fp) + Uziemienie
        "FP1": (0.34, 0.14, 0.10, 0.06),
        "FP2": (0.56, 0.14, 0.10, 0.06),
        "AFZ": (0.45, 0.19, 0.10, 0.06),  # AFz / GND
        # Linia czołowa (F)
        "F7": (0.19, 0.27, 0.10, 0.06),
        "F3": (0.33, 0.29, 0.10, 0.06),
        "FZ": (0.45, 0.30, 0.10, 0.06),
        "F4": (0.57, 0.29, 0.10, 0.06),
        "F8": (0.71, 0.27, 0.10, 0.06),
        "REF": (0.45, 0.40, 0.10, 0.06),  # Elektroda referencyjna
        # Linia centralna (C / T)
        "T3": (0.13, 0.49, 0.10, 0.06),
        "C3": (0.29, 0.49, 0.10, 0.06),
        "CZ": (0.45, 0.49, 0.10, 0.06),
        "C4": (0.61, 0.49, 0.10, 0.06),
        "T4": (0.77, 0.49, 0.10, 0.06),
        # Linia ciemieniowa (P / T)
        "T5": (0.20, 0.69, 0.10, 0.06),
        "P3": (0.33, 0.67, 0.10, 0.06),
        "PZ": (0.45, 0.67, 0.10, 0.06),
        "P4": (0.57, 0.67, 0.10, 0.06),
        "T6": (0.70, 0.69, 0.10, 0.06),
        # Linia potyliczna (O)
        "O1": (0.35, 0.82, 0.10, 0.06),
        "O2": (0.55, 0.82, 0.10, 0.06),
    }


    POSITIONS_1010 = {
    # Linia czołowo-centralna (FC) - brak FC5, FC6 w Twojej liście
    "FC3": (0.295, 0.38, 0.075, 0.045),
    "FC1": (0.380, 0.38, 0.075, 0.045),
    "FCZ": (0.462, 0.38, 0.075, 0.045),
    "FC2": (0.545, 0.38, 0.075, 0.045),
    "FC4": (0.630, 0.38, 0.075, 0.045),

    # Linia centralna (C) + Uszy / Wyrostki sutkowate (A1, A2)
    "A1":  (0.110, 0.49, 0.075, 0.045),
    "C5":  (0.210, 0.49, 0.075, 0.045),
    "C3":  (0.295, 0.49, 0.075, 0.045),
    "C1":  (0.380, 0.49, 0.075, 0.045),
    "CZ":  (0.462, 0.49, 0.075, 0.045),
    "C2":  (0.545, 0.49, 0.075, 0.045),
    "C4":  (0.630, 0.49, 0.075, 0.045),
    "C6":  (0.715, 0.49, 0.075, 0.045),
    "A2":  (0.815, 0.49, 0.075, 0.045),

    # Linia centralno-ciemieniowa (CP)
    "CP5": (0.210, 0.60, 0.075, 0.045),
    "CP3": (0.295, 0.60, 0.075, 0.045),
    "CP1": (0.380, 0.60, 0.075, 0.045),
    "CPZ": (0.462, 0.60, 0.075, 0.045),
    "CP2": (0.545, 0.60, 0.075, 0.045),
    "CP4": (0.630, 0.60, 0.075, 0.045),
    "CP6": (0.715, 0.60, 0.075, 0.045),

    # Linia ciemieniowa (P) - w Twojej liście tylko P1 oraz P2
    "P1":  (0.380, 0.71, 0.075, 0.045),
    "P2":  (0.545, 0.71, 0.075, 0.045),
    }

    def __init__(self, metadata_path): 
        
        config = super.__init__(metadata_path)
        
        print("===Initializing reading the data===")

        ### Reading the varaibles
        self.montage = config["montage"]
        self.higpass_cutoff = config["highpass_cutoff"]
        self.lowpass_cutoff = config["lowpass_cutoff"]
        self.project_name = config["project"]
        self.notch = config["notch"]
        self.save_path = config["save_path"]


        ### Mounting the data to reader
        file_paths = [self.project_name + file for file in [".obci.xml", ".obci.raw", ".obci.tag"]] 
        print(file_paths)
        mgr = ReadManager(file_paths[0], file_paths[1], file_paths[2])

        ### Extracting additional variables from reader
        self.sampling = float(mgr.get_param("sampling_frequency"))
        self.num_of_channels = int(mgr.get_param("number_of_channels"))
        self.channels_names = mgr.get_param("channels_names")
        self.tags = mgr.get_tags()
        syg = mgr.get_samples()

        # Formating data to dataframe
        self.data = pd.DataFrame.from_records(syg.T, columns=self.channels_names)
        
        print("Preforming…")


        
        # Preforming the montage
        print("- Montage:")
        for i in range(len(self.montage)):
            print(f"\t- {self.montage[i]}")

        if len(self.montage)==2:
            mean = ((self.data[self.montage[0]] + self.data[self.montage[1]])/2).to_numpy()
            new_syg = syg.T -  mean[:, np.newaxis]
        else:
            new_syg = syg.T - self.data[self.montage[0]].to_numpy()[:, np.newaxis]

        

        # Accouting for calibration and offset
        print("- Calibariotn:")
        print("Calibration parrams:")
        print("\t -Gain:")
        calibration_params = self.parse_calibration_params(file_paths[0])
        print(f"\t {calibration_params['gain']}")
        print("\t -Offset:")
        print(f"\t {calibration_params['offset']}")
        
        new_syg = new_syg* calibration_params["gain"].to_numpy()[np.newaxis, :]
        new_syg = new_syg+calibration_params["offset"].to_numpy()[np.newaxis, :]

        

        # Filtering
        print("-Filtering")
        print(f"\t- Lowpass: {self.lowpass_cutoff}")
        print(f"\t- Highpass: {self.higpass_cutoff}")

        
        b, a = ss.butter(
            N=4,
            Wn=[self.higpass_cutoff, self.lowpass_cutoff],
            btype="bandpass",
            fs=self.sampling,
        )


        syg_filtered = ss.filtfilt(b, a, new_syg, axis=0)


        # Preforimng notch
        if self.notch:
            print("-Filtering with Notch")

            b,a = ss.iirnotch(50, 5, self.sampling)
            syg_filtered = ss.filtfilt(b,a, syg_filtered, axis=0)
            
        self.data = pd.DataFrame.from_records(syg_filtered, columns=self.channels_names)
        times = np.arange(0,self.data.index.shape[0])/self.sampling

        self.data.index = times

        print("===DONE===")

        


    def read(self):
        return self.data
    

    
    def save(self):

        with open(os.path.join("eeg.npy"), "wb") as f:

            np.save(f, self.data)

        with open(os.path.join("tags.pkl"), "wb") as f:

            pickle.dump(self.tags, f)

    def parse_tag_xml(self, xml_path: str) -> pd.DataFrame:
        """Wczytuje plik XML z tagami i zwraca pd.DataFrame z kolumnami 'name' i 'position'.

        Index obiektu odpowiada kolejnym tagom w pliku.
        """
        # Wczytanie pliku XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        data = []

        # Znalezienie wszystkich elementów <tag> w strukturze pliku
        for tag in root.findall(".//tag"):
            name = tag.get("name")
            position = tag.get("position")

            # Konwersja rzutująca position na float (jeśli atrybut istnieje)
            position_val = float(position) if position is not None else None

            data.append({"name": name, "position": position_val})

        # Utworzenie DataFrame (domyślnie indeksem są kolejne liczby od 0)
        df = pd.DataFrame(data)

        return df


    def get_first_sample_timestamp(self,xml_path: str, key) -> float:
        """Wyciąga wartość z tagu <rs:firstSampleTimestamp> z podanego pliku XML.

        Zwraca wartość jako float.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Przestrzeń nazw (namespace) zdeklarowana w pliku XML
        namespaces = {"rs": "http://signalml.org/rawsignal"}

        # Wyszukanie tagu firstSampleTimestamp
        element = root.find("rs:" + key, namespaces)

        if element is not None and element.text:
            return float(element.text)
        else:
            raise ValueError(
                f"Nie znaleziono tagu <rs:firstSampleTimestamp> w pliku: {xml_path}"
            )
        

    def parse_calibration_params(self, xml_path: str) -> pd.DataFrame:
        """Wczytuje plik XML nagłówka EEG i zwraca DataFrame z kolumnami 'gain' oraz 'offset'.

        Indeksem są etykiety poszczególnych kanałów.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        namespaces = {"rs": "http://signalml.org/rawsignal"}

        # Wyciąganie etykiet kanałów
        labels = [
            label.text
            for label in root.findall(".//rs:channelLabels/rs:label", namespaces)
        ]

        # Wyciąganie wartości calibrationGain
        gains = [
            float(param.text)
            for param in root.findall(
                ".//rs:calibrationGain/rs:calibrationParam", namespaces
            )
        ]

        # Wyciąganie wartości calibrationOffset
        offsets = [
            float(param.text)
            for param in root.findall(
                ".//rs:calibrationOffset/rs:calibrationParam", namespaces
            )
        ]

        # Tworzenie DataFrame z indeksem ułożonym z etykiet kanałów
        df = pd.DataFrame({"gain": gains, "offset": offsets}, index=labels)

        return df
    

    @staticmethod
    def plot_eeg_on_head_image(
        df_data, channel_positions: dict = POSITIONS_1020, fs: float = 512.0
    ):
        """image_path: Ścieżka do pliku graficznego z głową (PNG/JPG) df_data: DataFrame z

        sygnałami EEG (kolumny to nazwy kanałów) channel_positions: Słownik z
        pozycjami {kanał: (x, y, width, height)} w ułamku wymiarów obrazka (0.0 -
        1.0)
        """
        # 1. Wczytanie obrazka tła
        CURRENT_DIR = Path(__file__).resolve().parent
        IMAGE_PATH = CURRENT_DIR / "images" / "img.jpeg"
        
        img = plt.imread(IMAGE_PATH)

        fig, ax = plt.subplots(figsize=(16, 16))

        # Wyświetlamy obrazek w głównych osiach (piksele jako współrzędne)
        ax.imshow(img)
        ax.axis("off")  # Ukrywamy główne osie

        img_height, img_width = img.shape[0], img.shape[1]

        # ax.set_xlim(img_width * 0.12, img_width * 0.88)
        # ax.set_ylim(img_height * 0.78, img_height * 0.12)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Pobieramy wymiary obrazka w pikselach
        img_height, img_width = img.shape[0], img.shape[1]

        # 2. Naniesienie każdego kanału na obrazek
        axes = {}
        for ch_name, pos in channel_positions.items():
            if ch_name in df_data.columns:
                x_rel, y_rel, w_rel, h_rel = pos

                x_px = x_rel * img_width
                y_px = y_rel * img_height
                w_px = w_rel * img_width
                h_px = h_rel * img_height

                ax_sub = ax.inset_axes(
                    [x_px, y_px, w_px, h_px], transform=ax.transData
                )
                axes[ch_name] = ax_sub

        return ax, axes, fig


    def permutaion_test_per_channel(data, ch_name):
        """
        Fucntion that computes the statistical test

        Parameters 
        ----------
        - data : pd.DataFrame
            dataframe with chanels as columns
        
        """


    @staticmethod
    def plot_eeg_on_head_image_1010_old(
        df_data, channel_positions: dict = POSITIONS_1010, fs: float = 512.0
    ):
        """
        df_data: DataFrame z sygnałami EEG (kolumny to nazwy kanałów)
        channel_positions: Słownik z pozycjami {kanał: (x, y, width, height)}
                           w ułamku wymiarów obrazka (0.0 - 1.0)
        """
        # 1. Wczytanie obrazka tła
        CURRENT_DIR = Path(__file__).resolve().parent
        IMAGE_PATH = CURRENT_DIR / "images" / "img.jpeg"

        img = plt.imread(IMAGE_PATH)

        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(img)
        ax.axis("off")

        img_height, img_width = img.shape[0], img.shape[1]

        # Poszerzony zakres X, aby zmieścić A1 (0.11) i A2 (0.815 + 0.075 = 0.89)
        ax.set_xlim(img_width * 0.08, img_width * 0.92)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # 2. Naniesienie każdego kanału na obrazek
        axes = {}
        for ch_name, pos in channel_positions.items():
            x_rel, y_rel, w_rel, h_rel = pos

            x_px = x_rel * img_width
            y_px = y_rel * img_height
            w_px = w_rel * img_width
            h_px = h_rel * img_height

            ax_sub = ax.inset_axes(
                [x_px, y_px, w_px, h_px], transform=ax.transData
            )

            if ch_name in df_data.columns:
                axes[ch_name] = ax_sub
            else:
                # Brak kanału w danych -> pusty wykres (tylko subtelna ramka i etykieta)
                ax_sub.set_xticks([])
                ax_sub.set_yticks([])
                ax_sub.set_title(ch_name, fontsize=8, color="gray")
                ax_sub.patch.set_alpha(0.3)

        return ax, axes, fig
    
