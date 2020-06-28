from xml.etree import ElementTree

import pandas as pd

import glob
import os
import shutil
import subprocess

import music21

jsymbolic_feature_values_fname = (
    "/Users/bgeelen/jSymbolic_2_2_user/acexmlfeaturevaluesoutput.xml"
)
jsymbolic_jar_location = "/Users/bgeelen/jSymbolic_2_2_user/jSymbolic2.jar"
jrp_dataset_krn_root = "/Users/bgeelen/Data/josquin/source"
jrp_dataset_midi_root = "/Users/bgeelen/Data/josquin/midi"

#%%


def convert_file_krn_to_midi(in_fname, out_fname):
    try:
        parsed = music21.converter.parse(in_fname)
        parsed.write("midi", out_fname)
    except Exception as e:
        print(f"Exception with file {in_fname}:\n{e}")


def convert_dir_krn_to_midi(in_dir, out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(in_dir, out_dir)

    all_fnames = glob.glob(os.path.join(out_dir, "**", "*.krn"), recursive=True)
    for fname in all_fnames:
        out_fname = f"{fname[:-4]}.mid"
        convert_file_krn_to_midi(fname, out_fname)
        os.remove(fname)


if __name__ == "__main__":
    if not os.path.exists(jsymbolic_feature_values_fname):
        print("Converting krn files to midi files...")
        convert_dir_krn_to_midi(jrp_dataset_krn_root, jrp_dataset_midi_root)

        print("Extracting features using jsymbolic...")
        print("This process could take multiple hours.")
        #%%
        jsymbolic_feature_definitions_fname = os.path.join(
            os.path.dirname(jsymbolic_feature_values_fname),
            "acexmlfeaturedefinitionsoutput.xml",
        )
        #%%
        result = subprocess.run(
            [
                "java",
                "-Xmx50G",
                "-jar",
                jsymbolic_jar_location,
                "-arff",
                jrp_dataset_midi_root,
                jsymbolic_feature_values_fname,
                jsymbolic_feature_definitions_fname,
            ]
        )
    #%%
    feature_definitions = ElementTree.parse(jsymbolic_feature_definitions_fname)
    feature_values = ElementTree.parse(jsymbolic_feature_values_fname)

#%%

def feature_values_file_to_dataframe(feature_values_fname):
    feature_values = ElementTree.parse(feature_values_fname)

    as_dict = {
        item.findall("data_set_id")[0].text: {
            feature.find("name").text: feature.find("v").text
            for feature in item.findall("feature")
        }
        for item in feature_values.findall("data_set")
    }

    df = pd.DataFrame(as_dict).T

    return df

df = feature_values_file_to_dataframe('feature_values.xml')
