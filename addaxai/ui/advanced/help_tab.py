"""Help tab content for AddaxAI — HyperlinkManager and write_help_tab."""

import webbrowser
from functools import partial
from tkinter import END, INSERT, CURRENT, NORMAL, DISABLED
from typing import Any, Dict, Optional, Tuple

from addaxai.i18n import t, lang_idx as i18n_lang_idx

# create hyperlinks (thanks marvin from GitHub)
class HyperlinkManager:
    def __init__(self, text: Any, green_primary: str = "#00A86B") -> None:
        self.text = text
        self.text.tag_config("hyper", foreground=green_primary, underline=1)
        self.text.tag_bind("hyper", "<Enter>", self._enter)
        self.text.tag_bind("hyper", "<Leave>", self._leave)
        self.text.tag_bind("hyper", "<Button-1>", self._click)
        self.reset()

    def reset(self) -> None:
        self.links: Dict[str, Any] = {}

    def add(self, action: Any) -> Tuple[str, str]:
        tag = "hyper-%d" % len(self.links)
        self.links[tag] = action
        return "hyper", tag

    def _enter(self, event: Any) -> None:
        self.text.configure(cursor="hand2")

    def _leave(self, event: Any) -> None:
        self.text.configure(cursor="")

    def _click(self, event: Any) -> None:
        for tag in self.text.tag_names(CURRENT):
            if tag[:6] == "hyper-":
                self.links[tag]()
                return



def write_help_tab(help_text_widget: Any, hyperlink: Any, text_font: str = "TkDefaultFont", scroll: Optional[Any] = None) -> None:
    line_number = 1 

    # intro sentence
    help_text_widget.insert(END, ["Below you can find detailed documentation for each setting. If you have any questions, feel free to contact me on ",
                           "A continuación encontrarás documentación detallada sobre cada ajuste. Si tienes alguna pregunta, no dudes en ponerte en contacto conmigo en ",
                           "Ci-dessous, vous trouverez la documentation détaillée pour chaque paramètre. Si vous avez des questions, n'hésitez pas à me contacter (anglais) à l'adresse "][i18n_lang_idx()])
    help_text_widget.insert(INSERT, "peter@addaxdatascience.com", hyperlink.add(partial(webbrowser.open, "mailto:peter@addaxdatascience.com")))
    help_text_widget.insert(END, [" or raise an issue on the ", " o plantear una incidencia en ", " ou à rapporter un incident sur la "][i18n_lang_idx()])
    help_text_widget.insert(INSERT, t('github_page'), hyperlink.add(partial(webbrowser.open, "https://github.com/PetervanLunteren/AddaxAI/issues")))
    help_text_widget.insert(END, ".\n\n")
    help_text_widget.tag_add('intro', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # first step
    help_text_widget.insert(END, t('fst_step') + "\n")
    help_text_widget.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.insert(END, t('browse') + "\n")
    help_text_widget.insert(END, ["Here you can browse for a folder which contains images and/or video\'s. The model will be deployed on this directory, as well as the post-processing analyses.\n\n",
                           "Aquí puede buscar una carpeta que contenga imágenes y/o vídeos. El modelo se desplegará en este directorio, así como los análisis de post-procesamiento.\n\n",
                           "Ici vous pouvez spécifier un dossier contenant des images et/ou des vidéos. La détection, l'identifications et les analyses de post-traitement s'effectueront sur le contenu de ce répertoire.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # second step
    help_text_widget.insert(END, t('snd_step') + "\n")
    help_text_widget.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # det model
    help_text_widget.insert(END, f"{t('lbl_model')}\n")
    help_text_widget.insert(END, [
        "AddaxAI uses a combination of a detection model and a classification model to identify animals. The detection model will locate the animal, whereas the "
        "classification model will identify which species the animal belongs to. Here, you can select the detection model that you want to use. If the dropdown "
        "option 'Custom model' is selected, you will be prompted to select a custom YOLOv5 model file. The preloaded 'MegaDetector' models detect animals, people, "
        "and vehicles in camera trap imagery. It does not identify the animals; it just finds them. Version 5a and 5b differ only in their training data. Each may perform "
        "slightly better depending on your specific dataset, so feel free to compare. If you don’t know where to start, stick with the default 'MegaDetector 5a'. "
        "Newer detection models are also available. 'MDv1000-redwood' is a newer, larger model designed to improve detection accuracy and shows very high recall, but "
        "it still needs community testing. Use it if you're willing to explore cutting-edge performance. 'MDv1000-spruce' is a smaller, faster model, but only use it if speed is a key requirement "
        "and you're confident about its trade-offs. In general, speed is rarely a bottleneck anymore, since even modest hardware can process tens of thousands of images per day. "
        "More info about MegaDetector models ",
        
        "AddaxAI utiliza una combinación de un modelo de detección y un modelo de clasificación para identificar animales. El modelo de detección localizará al animal, mientras que el modelo de "
        "clasificación identificará a qué especie pertenece. Aquí puede seleccionar el modelo de detección que desea utilizar. Si selecciona la opción desplegable 'Modelo personalizado', se le "
        "pedirá que seleccione un archivo de modelo YOLOv5 personalizado. Los modelos 'MegaDetector' precargados detectan animales, personas y vehículos en imágenes de cámaras trampa. "
        "No identifican a los animales; solo los encuentran. Las versiones 5a y 5b difieren únicamente en sus datos de entrenamiento. Uno puede funcionar ligeramente mejor que el otro dependiendo de sus datos. "
        "Si no está seguro, utilice el modelo predeterminado 'MegaDetector 5a'. "
        "También hay modelos nuevos disponibles. 'MDv1000-redwood' es un modelo más grande diseñado para mejorar la precisión y tiene un excelente recall, aunque aún requiere más pruebas por la comunidad. "
        "Úselo si quiere probar lo más avanzado. 'MDv1000-spruce' es un modelo más pequeño y rápido, pero sólo se recomienda si la velocidad es una prioridad clara y conoce sus limitaciones. En general, la velocidad rara vez es un cuello de botella, "
        "ya que incluso hardware modesto puede procesar decenas de miles de imágenes por día. "
        "Más información sobre los modelos MegaDetector ",

        "AddaxAI utilise une combinaison de modèles de détection et de classification pour identifier les animaux. Le modèle de détection localise l'animal, tandis que le modèle de classification "
        "détermine à quelle espèce appartient l'animal. Vous pouvez sélectionner ici le modèle de détection souhaité. Si vous sélectionnez l’option « Modèle personnalisé », vous devrez choisir un fichier YOLOv5 personnalisé. "
        "Les modèles 'MegaDetector' préchargés détectent les animaux, les personnes et les véhicules dans les images de pièges photographiques. Ils ne reconnaissent pas les espèces. Les versions 5a et 5b ne diffèrent que par leurs données d'entraînement. "
        "L’un peut être légèrement meilleur que l’autre selon vos données. Si vous hésitez, choisissez le modèle par défaut 'MegaDetector 5a'. "
        "Des modèles plus récents sont également disponibles. 'MDv1000-redwood' est un modèle plus grand conçu pour une meilleure précision et montre un très bon rappel, mais il doit encore être testé par la communauté. "
        "Utilisez-le si vous souhaitez expérimenter des performances de pointe. 'MDv1000-spruce' est plus petit et plus rapide, mais n’est recommandé que si la vitesse est une exigence importante et que vous en acceptez les compromis. "
        "De manière générale, la vitesse n'est presque jamais un facteur limitant, car même du matériel modeste peut traiter des dizaines de milliers d'images par jour. "
        "Plus d'informations sur les modèles MegaDetector "
    ][i18n_lang_idx()])
    help_text_widget.insert(INSERT, t('here'), hyperlink.add(partial(webbrowser.open, "https://github.com/ecologize/CameraTraps/blob/main/megadetector.md#megadetector-v50-20220615")))
    help_text_widget.insert(END, ".\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 2

    # cls model
    help_text_widget.insert(END, f"{t('lbl_cls_model')}\n")
    help_text_widget.insert(END, ["AddaxAI uses a combination of a detection model and a classification model to identify animals. The detection model will locate the animal, whereas the "
                           "classification model will identify which species the animal belongs to. Here, you can select the classification model that you want to use. Each "
                           "classification model is developed for a specific area. Explore which model suits your data best, but please note that models developed for other biomes "
                           "or projects do not necessarily perform equally well in other ecosystems. Always investigate the model’s accuracy on your data before accepting any results.", 
                           "AddaxAI utiliza una combinación de un modelo de detección y un modelo de clasificación para identificar animales. El modelo de detección localizará al "
                           "animal, mientras que el modelo de clasificación identificará a qué especie pertenece el animal. Aquí puede seleccionar el modelo de clasificación que desea "
                           "utilizar. Cada modelo de clasificación se desarrolla para un área específica. Explore qué modelo se adapta mejor a sus datos, pero tenga en cuenta que los "
                           "modelos desarrollados para otros biomas o proyectos no funcionan necesariamente igual de bien en otros ecosistemas. Investiga siempre la precisión del modelo"
                           " en tus datos antes de aceptar cualquier resultado.",
                           "AddaxAI utilise une combinaison de modèles de détection et de classification pour identifier les animaux. Le modèle de détection localise l'animal, tandis "
                           "que le modèle de classification identifie l'espèce à laquelle il appartient. Vous pouvez sélectionner ici le modèle de classification que vous souhaitez "
                           "utiliser. Chaque modèle de classification est développé pour une zone spécifique. Découvrez quel modèle est le plus adapté à vos données, mais veuillez "
                           "noter que les modèles développés pour d'autres biomes ou projets ne sont pas nécessairement aussi performants dans d'autres écosystèmes. Vérifiez toujours "
                           "la précision du modèle sur vos données avant d'accepter les résultats."][i18n_lang_idx()])
    help_text_widget.insert(END, "\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # cls model info
    help_text_widget.insert(END, f"{t('lbl_model_info')}\n")
    help_text_widget.insert(END, ["This will open a window with model information.", "Esto abrirá una ventana con información sobre el modelo.", "Ceci ouvrira une fenêtre avec des informations sur le modèle."][i18n_lang_idx()])
    help_text_widget.insert(END, "\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # cls spp selection
    help_text_widget.insert(END, f"{t('lbl_choose_classes')}\n")
    help_text_widget.insert(END, ["Here, you can select and deselect the animals categories that are present in your project"
                          " area. If the animal category is not selected, it will be excluded from the results. The "
                          "category list will update according to the model selected.", "Aquí puede seleccionar y anular"
                          " la selección de las categorías de animales presentes en la zona de su proyecto. Si la "
                          "categoría de animales no está seleccionada, quedará excluida de los resultados. La lista de "
                          "categorías se actualizará según el modelo seleccionado.",
                          "Ici, vous pouvez sélectionner et désélectionner les catégories d'animaux présentes dans "
                          "la zone d'étude de votre projet. Si la catégorie d'animal n'est pas sélectionnée, elle sera "
                          "exclue des résultats. La liste des catégories sera mise à jour en fonction du modèle sélectionné."][i18n_lang_idx()])
    help_text_widget.insert(END, "\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # threshold to classify detections
    help_text_widget.insert(END, f"{t('lbl_cls_detec_thresh')}\n")
    help_text_widget.insert(END, ["AddaxAI uses a combination of a detection model and a classification model to identify animals. The detection model will locate "
                           "the animal, whereas the classification model will identify which species the animal belongs to. This confidence threshold defines "
                           "which animal detections will be passed on to the classification model for further identification.", "AddaxAI utiliza una "
                           "combinación de un modelo de detección y un modelo de clasificación para identificar a los animales. El modelo de detección "
                           "localizará al animal, mientras que el modelo de clasificación identificará a qué especie pertenece el animal. Este umbral de "
                           "confianza define qué animales detectados se pasarán al modelo de clasificación para su posterior identificación.",
                           "AddaxAI utilise une combinaison de modèles de détection et de classification pour identifier les animaux. Le modèle de détection "
                           "localise l'animal, tandis que le modèle de classification identifie son espèce. Ce seuil de confiance définit les détections "
                           "d'animaux qui seront transmises au modèle de classification pour une identification plus approfondie."][i18n_lang_idx()])
    help_text_widget.insert(END, "\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # threshold to classify detections
    help_text_widget.insert(END, f"{t('lbl_cls_class_thresh')}\n")
    help_text_widget.insert(END, ["AddaxAI uses a combination of a detection model and a classification model to identify animals. The detection model will locate "
                           "the animal, whereas the classification model will identify which species the animal belongs to. This confidence threshold defines "
                           "which animal identifications will be accepted.", "AddaxAI utiliza una combinación de un modelo de detección y un modelo de "
                           "clasificación para identificar a los animales. El modelo de detección localizará al animal, mientras que el modelo de clasificación"
                           " identificará a qué especie pertenece el animal. Este umbral de confianza define qué identificaciones de animales se aceptarán.",
                           "AddaxAI utilise une combinaison de modèles de détection et de classification pour identifier les animaux. Le modèle de détection "
                           "localise l'animal, tandis que le modèle de classification identifie son espèce. Ce seuil de confiance définit les classifications "
                           "d'animaux qui seront considérées valides."][i18n_lang_idx()])
    help_text_widget.insert(END, "\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # smooth results
    help_text_widget.insert(END, f"{t('lbl_smooth_cls_animal')}\n")
    help_text_widget.insert(END, ["Sequence smoothing averages confidence scores across detections within a sequence to reduce noise. This improves accuracy by "
                           "providing more stable results by combining information over multiple images. Note that it assumes a single species per "
                           "sequence and should therefore only be used if multi-species sequences are rare. It does not affect detections of vehicles or "
                           "people alongside animals.", "El suavizado de secuencias promedia las puntuaciones de confianza entre detecciones dentro de "
                           "una secuencia para reducir el ruido. Esto mejora la precisión al proporcionar resultados más estables mediante la combinación"
                           " de información de múltiples imágenes. Tenga en cuenta que supone una única especie por secuencia y, por lo tanto, sólo debe "
                           "utilizarse si las secuencias multiespecie son poco frecuentes. No afecta a las detecciones de vehículos o personas junto a "
                           "animales.",
                           "Le lissage fait la moyenne des scores de confiance des détections au sein d'une séquence afin de réduire les "
                           "aberrations statistiques. Cela améliore la précision en fournissant des résultats plus stables grâce à la combinaison des "
                           "informations de plusieurs images. Notez que ce lissage suppose une seule espèce par séquence et ne doit donc être utilisé "
                           "que si les séquences multi-espèces sont rares. Il n'affecte pas les détections de véhicules ou de personnes à proximité "
                           "d'animaux."][i18n_lang_idx()])
    help_text_widget.insert(END, "\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # fallback to higher taxonomy if uncertain
    help_text_widget.insert(END, f"{t('lbl_tax_fallback')}\n")
    help_text_widget.insert(END, [
        "If enabled, the model will automatically fall back to a higher taxonomic level (e.g., genus or family) when its confidence at the species level is low. "
        "This can improve overall prediction accuracy by avoiding uncertain species-level classifications. Note that some categories may not have species-level predictions at all — "
        "for example, if the model was trained on a broader class like 'bird', it will never predict individual bird species. "
        "This option is only available in models that have been adjusted to support taxonomic fallback.",
        
        "Si está activado, el modelo recurrirá automáticamente a un nivel taxonómico superior (por ejemplo, género o familia) cuando su confianza en el nivel de especie sea baja. "
        "Esto puede mejorar la precisión general al evitar clasificaciones inciertas a nivel de especie. Tenga en cuenta que algunas categorías no tienen predicciones a nivel de especie — "
        "por ejemplo, si el modelo fue entrenado en una clase amplia como 'ave', nunca predecirá especies individuales de aves. "
        "Esta opción solo está disponible en modelos que han sido ajustados para admitir el retroceso taxonómico.",

        "Si cette option est activée, le modèle reviendra automatiquement à un niveau taxonomique supérieur (par exemple, genre ou famille) lorsque son niveau de confiance"
        "de l'espèce est faible. Cela peut améliorer la précision globale des prédictions en évitant les classifications incertaines au niveau de l'espèce. Notez que certaines "
        "catégories peuvent ne pas avoir de prédictions au niveau de l'espèce du tout — par exemple, si le modèle a été entraîné sur une classe plus large comme « oiseau », "
        "il ne prédira jamais les espèces d'oiseaux individuelles. Cette option est uniquement disponible dans les modèles qui ont été ajustés pour prendre en charge l'aggrégation "
        "taxonomique."
    ][i18n_lang_idx()])
    help_text_widget.insert(END, "\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 2

    # prediction level
    help_text_widget.insert(END, f"{t('lbl_tax_levels')}\n")
    help_text_widget.insert(END, [
        "This setting allows you to control the granularity of the model's predictions. By default, the model will automatically choose the most appropriate "
        "taxonomic level (e.g., species, genus, or family) depending on its confidence. You can also choose to force predictions to a specific level, "
        "such as always predicting at the family or genus level, if available. This can be useful for applications where fine-grained classification is not needed or where high-level consistency is preferred. "
        "You can also restrict predictions to only those categories that had a minimum number of training samples (e.g., ≥ 10,000). This helps reduce errors caused by underrepresented categories, "
        "but may result in some predictions being skipped entirely if they don’t meet the threshold. "
        "Note that the availability of certain taxonomic levels (e.g., class, genus, species) depends on how the model was trained. If a level is not available for a given detection, "
        "the model will fall back to the closest broader category.",
        
        "Esta opción le permite controlar el nivel de detalle de las predicciones del modelo. Por defecto, el modelo elegirá automáticamente el nivel taxonómico más adecuado "
        "(por ejemplo, especie, género o familia), en función de su confianza y de cómo se entrenó el modelo. También puede optar por forzar las predicciones a un nivel específico, "
        "como predecir siempre a nivel de familia o género, si están disponibles. Esto puede ser útil en aplicaciones en las que no se necesita una clasificación precisa "
        "o se prefiere una consistencia a un nivel superior. "
        "También puede restringir las predicciones a categorías que hayan tenido un número mínimo de muestras de entrenamiento (por ejemplo, ≥ 10,000). Esto ayuda a reducir errores "
        "causados por categorías poco representadas, pero puede dar lugar a que algunas predicciones se omitan si no cumplen ese umbral. "
        "Tenga en cuenta que la disponibilidad de ciertos niveles taxonómicos (por ejemplo, clase, género, especie o grupo de edad y sexo) depende de cómo se entrenó el modelo. "
        "Si un nivel no está disponible para una detección determinada, el modelo recurrirá al nivel más amplio disponible.",

        "Ce paramètre vous permet de contrôler la granularité des prédictions du modèle. Par défaut, le modèle choisit automatiquement le niveau taxonomique le plus approprié (par exemple, espèce, genre "
        "ou famille) en fonction de sa fiabilité. Vous pouvez également forcer les prédictions à un niveau spécifique, par exemple en prédisant toujours au niveau de la famille ou du genre, si disponible."
        " Cela peut être utile pour les applications où une classification fine n'est pas nécessaire ou où une cohérence de haut niveau est privilégiée. Vous pouvez également restreindre les prédictions aux "
        "catégories ayant un nombre minimum d'échantillons d'apprentissage (par exemple, ≥ 10 000). Cela permet de réduire les erreurs causées par des catégories sous-représentées, mais peut entraîner "
        "l'abandon complet de certaines prédictions si elles n'atteignent pas le seuil. Notez que la disponibilité de certains niveaux taxonomiques (par exemple, classe, genre, espèce) dépend de la manière "
        "dont le modèle a été entraîné. Si un niveau n'est pas disponible pour une détection donnée, le modèle reviendra à la catégorie la plus proche."
    ][i18n_lang_idx()])
    help_text_widget.insert(END, "\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 2


    # exclude subs
    help_text_widget.insert(END, f"{t('lbl_exclude_subs')}\n")
    help_text_widget.insert(END, ["By default, AddaxAI will recurse into subdirectories. Select this option if you want to ignore the subdirectories and process only"
                           " the files directly in the chosen folder.\n\n", "Por defecto, AddaxAI buscará en los subdirectorios. Seleccione esta opción si "
                           "desea ignorar los subdirectorios y procesar sólo los archivos directamente en la carpeta elegida.\n\n",
                           "Par défaut, AddaxAI effectue une récursion dans les sous-répertoires. Sélectionnez cette option si vous souhaitez ignorer les "
                           "sous-répertoires et traiter uniquement les fichiers directement dans le dossier choisi.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # exclude detections
    help_text_widget.insert(END, f"{t('lbl_use_custom_img_size_for_deploy')} / {t('lbl_image_size_for_deploy')}\n")
    help_text_widget.insert(END, ["AddaxAI will resize the images before they get processed. AddaxAI will by default resize the images to 1280 pixels. "
                    "Deploying a model with a lower image size will reduce the processing time, but also the detection accuracy. Best results are obtained if you use the"
                    " same image size as the model was trained on. If you trained a model in AddaxAI using the default image size, you should set this value to 640 for "
                    "the YOLOv5 models. Use the default for the MegaDetector models.\n\n",
                    "AddaxAI redimensionará las imágenes antes de procesarlas. Por defecto, AddaxAI redimensionará las imágenes a 1280 píxeles. Desplegar un modelo "
                    "con un tamaño de imagen inferior reducirá el tiempo de procesamiento, pero también la precisión de la detección. Los mejores resultados se obtienen "
                    "si se utiliza el mismo tamaño de imagen con el que se entrenó el modelo. Si ha entrenado un modelo en AddaxAI utilizando el tamaño de imagen por "
                    "defecto, debe establecer este valor en 640 para los modelos YOLOv5. Utilice el valor por defecto para los modelos MegaDetector.\n\n",
                    "AddaxAI redimensionne les images avant leur traitement. Par défaut, AddaxAI les redimensionne à 1280 pixels. Déployer un modèle avec une taille "
                    "d'image inférieure réduira le temps de traitement, mais aussi la précision de la détection. De meilleurs résultats sont obtenus en utilisant la "
                    "même taille d'image que celle utilisée pour l'entraînement du modèle. Si vous avez entraîné un modèle dans AddaxAI avec la taille d'image par "
                    "défaut, vous devez définir cette valeur sur 640 pour les modèles YOLOv5. Utilisez la valeur par défaut pour les modèles MegaDetector.\n\n"
                    ][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # use absolute paths
    help_text_widget.insert(END, f"{t('lbl_abs_paths')}\n")
    help_text_widget.insert(END, ["By default, the paths in the output file are relative (i.e. 'image.jpg') instead of absolute (i.e. '/path/to/some/folder/image.jpg'). This "
                    "option will make sure the output file contains absolute paths, but it is not recommended. Third party software (such as ",
                    "Por defecto, las rutas en el archivo de salida son relativas (es decir, 'imagen.jpg') en lugar de absolutas (es decir, '/ruta/a/alguna/carpeta/"
                    "imagen.jpg'). Esta opción se asegurará de que el archivo de salida contenga rutas absolutas, pero no se recomienda. Software de terceros (como ",
                    "Par défaut, les chemins du fichier de sortie sont relatifs (par exemple, « image.jpg ») et non absolus (par exemple, "
                    "« /chemin/vers/un/dossier/image.jpg »). Cette option garantit que le fichier de sortie contient des chemins absolus, mais elle n'est pas "
                    "recommandée. Les logiciels tiers (tel que « "][i18n_lang_idx()])
    help_text_widget.insert(INSERT, "Timelapse", hyperlink.add(partial(webbrowser.open, "https://timelapse.ucalgary.ca/")))
    help_text_widget.insert(END, [") will not be able to read the output file if the paths are absolute. Only enable this option if you know what you are doing. More information"
                    " how to use Timelapse in conjunction with MegaDetector, see the ",
                    ") no serán capaces de leer el archivo de salida si las rutas son absolutas. Solo active esta opción si sabe lo que está haciendo. Para más información"
                    " sobre cómo utilizar Timelapse junto con MegaDetector, consulte ",
                    " », de Saul Greenberg, University of Calgary) ne pourra pas lire le fichier de sortie si les chemins sont absolus. N'activez cette option que si vous savez ce que vous faites. Pour plus "
                    "d'informations, consultez la section « Comment utiliser Timelapse avec MegaDetector ? ». "][i18n_lang_idx()])
    help_text_widget.insert(INSERT, ["Timelapse Image Recognition Guide", "la Guía de Reconocimiento de Imágenes de Timelapse", "Guide de reconnaissance avec Timelapse (anglais)"
    ][i18n_lang_idx()], hyperlink.add(partial(webbrowser.open, "https://timelapse.ucalgary.ca/wp-content/uploads/Guides/TimelapseImageRecognitionGuide.pdf")))
    help_text_widget.insert(END, ".\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # use checkpoints
    help_text_widget.insert(END, f"{t('lbl_use_checkpnts')}\n")
    help_text_widget.insert(END, ["This is a functionality to save results to checkpoints intermittently, in case a technical hiccup arises. That way, you won't have to restart"
                    " the entire process again when the process is interrupted.\n\n",
                    "Se trata de una funcionalidad para guardar los resultados en puntos de control de forma intermitente, en caso de que surja un contratiempo técnico. "
                    "De esta forma, no tendrás que reiniciar todo el proceso de nuevo cuando éste se interrumpa.\n\n",
                    "Cette fonctionnalité permet d'enregistrer les résultats dans des points de contrôle de manière intermittente, en cas de problème technique. Ainsi, "
                    "vous n'aurez pas à redémarrer le processus en cas d'interruption.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # checkpoint frequency
    help_text_widget.insert(END, f"{t('lbl_checkpoint_freq')}\n")
    help_text_widget.insert(END, ["Fill in how often you want to save the results to checkpoints. The number indicates the number of images after which checkpoints will be saved."
                    " The entry must contain only numeric characters.\n\n",
                    "Introduzca la frecuencia con la que desea guardar los resultados en los puntos de control. El número indica el número de imágenes tras las cuales se "
                    "guardarán los puntos de control. La entrada debe contener sólo caracteres numéricos.\n\n",
                    "Indiquez la fréquence à laquelle vous souhaitez enregistrer les résultats dans les points de contrôle. Le nombre indique le nombre d'images après "
                    "lequel les points de contrôle seront enregistrés. L'entrée doit contenir uniquement des caractères numériques.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # continue from checkpoint
    help_text_widget.insert(END, f"{t('lbl_cont_checkpnt')}\n")
    help_text_widget.insert(END, ["Here you can choose to continue from the last saved checkpoint onwards so that the algorithm can continue where it left off. Checkpoints are"
                    " saved into the main folder and look like 'checkpoint_<timestamp>.json'. When choosing this option, it will search for a valid"
                    " checkpoint file and prompt you if it can't find it.\n\n",
                    "Aquí puede elegir continuar desde el último punto de control guardado para que el algoritmo pueda continuar donde lo dejó. Los puntos de control se "
                    "guardan en la carpeta principal y tienen el aspecto 'checkpoint_<fecha y hora>.json'. Al elegir esta opción, se buscará un archivo de punto de control "
                    "válido y se le preguntará si no puede encontrarlo.\n\n",
                    "Ici, vous pouvez choisir de continuer à partir du dernier point de contrôle enregistré afin que l'algorithme puisse reprendre là où il s'est arrêté. "
                    "Les points de contrôle sont enregistrés dans le dossier principal et ressemblent à « checkpoint_<horodatage>.json ». Lorsque vous choisissez cette "
                    "option, l'algorithme recherche un fichier de point de contrôle valide et vous avertit s'il ne le trouve pas.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # don't process every frame
    help_text_widget.insert(END, f"{t('lbl_not_all_frames')}\n")
    help_text_widget.insert(END,["When processing every frame of a video, it can take a long time to finish. Here, you can specify whether you want to analyse only a selection of frames."
                    f" At '{t('lbl_nth_frame')}' you can specify how many frames you want to be analysed.\n\n",
                     "Procesar todos los fotogramas de un vídeo puede llevar mucho tiempo. Aquí puede especificar si desea analizar sólo una selección de fotogramas. "
                    f"En '{t('lbl_nth_frame')}' puedes especificar cuántos fotogramas quieres que se analicen.\n\n",
                    "Le traitement de chaque image d'une vidéo peut prendre un certain temps. Vous pouvez ici spécifier si vous souhaitez analyser uniquement une sélection d'images."
                    f" L'option « {t('lbl_nth_frame')} » permet de spécifier combien de trames (images) vous souhaitez analyser.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # analyse every nth frame
    help_text_widget.insert(END, f"{t('lbl_nth_frame')}\n")
    help_text_widget.insert(END, ["Specify the frame sampling rate you'd like to use. For example, entering '1' will process one frame per second. Typically, sampling one frame per second is sufficient and can significantly reduce processing time. The exact time savings depend on the video's frame rate. Most camera traps record at 30 frames per second, meaning this approach can reduce processing time by 97% compared to processing every frame.\n\n",
                    "Especifica la tasa de muestreo de fotogramas que deseas utilizar. Por ejemplo, ingresar '1' procesará un fotograma por segundo. Generalmente, muestrear un fotograma por segundo es suficiente y puede reducir significativamente el tiempo de procesamiento. El ahorro exacto de tiempo depende de la tasa de fotogramas del video. La mayoría de las cámaras trampa graban a 30 fotogramas por segundo, lo que significa que este enfoque puede reducir el tiempo de procesamiento aproximadamente en un 97% en comparación con procesar todos los fotogramas.\n\n",
                    "Spécifiez la fréquence d'échantillonnage d'images que vous souhaitez utiliser. Par exemple, saisir « 1 » traitera une image par seconde. En général, un échantillonnage d'une image par seconde est suffisant et peut réduire considérablement le temps de traitement. Le gain de temps exact dépend de la fréquence d'images de la vidéo. La plupart des pièges photographiques enregistrent à 30 images par seconde, ce qui signifie que cette approche peut réduire le temps de traitement de 97 % par rapport au traitement de chaque image.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # third step
    help_text_widget.insert(END, t('trd_step') + "\n")
    help_text_widget.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # human verification
    help_text_widget.insert(END, f"{t('lbl_hitl_main')}\n")
    help_text_widget.insert(END, ["This feature lets you verify the results of the model. You can use it to create training data or to double-check the results. When starting a new "
                           "session, you will first be directed to a window where you can select which images you would like to verify. For instance, someone might be only "
                           "interested in creating training data for 'class A' to unbalance his training dataset or only want to double-check detections with medium-sure "
                           "confidences. After you have selected the images, you will be able to verify them. After having verified all selected images, you will be prompted"
                           " if you want to create training data. If you do, the selected images and their associated annotation files will get a unique name and be either "
                           "moved or copied to a folder of your choice. This is particularly handy when you want to create training data since, for training, all files must "
                           "be in one folder. This way, the files will be unique, and you won't have replacement problems when adding the files to your existing training data. "
                           "You can also skip the training data and just continue to post-process the verified results. Not applicable to videos.\n\n",
                           "Esta característica le permite verificar los resultados del modelo. Puedes usarlo para crear datos de entrenamiento o para verificar los resultados. "
                           "Al iniciar una nueva sesión, primero se le dirigirá a una ventana donde podrá seleccionar qué imágenes desea verificar. Por ejemplo, alguien podría "
                           "estar interesado únicamente en crear datos de entrenamiento para la 'clase A' para desequilibrar su conjunto de datos de entrenamiento o simplemente "
                           "querer verificar las detecciones con confianzas medias-seguras. Una vez que hayas seleccionado las imágenes, podrás verificarlas. Después de haber "
                           "verificado todas las imágenes seleccionadas, se le preguntará si desea crear datos de entrenamiento. Si lo hace, las imágenes seleccionadas y sus "
                           "archivos de anotaciones asociados obtendrán un nombre único y se moverán o copiarán a una carpeta de su elección. Esto es particularmente útil cuando"
                           " desea crear datos de entrenamiento ya que, para el entrenamiento, todos los archivos deben estar en una carpeta. De esta manera, los archivos serán "
                           "únicos y no tendrás problemas de reemplazo al agregar los archivos a tus datos de entrenamiento existentes. También puedes omitir los datos de "
                           "entrenamiento y simplemente continuar con el posprocesamiento de los resultados verificados. No aplicable a vídeos.\n\n",
                           "Cette fonctionnalité vous permet de vérifier les résultats du modèle. Vous pouvez l'utiliser pour créer des données d'entraînement ou pour revérifier "
                           "les résultats. Lorsque vous démarrez une nouvelle session, vous serez d'abord redirigé vers une fenêtre vous permettant de sélectionner les images à "
                           "vérifier. Par exemple, quelqu'un pourrait souhaiter créer uniquement des données d'entraînement pour la « classe A » afin de déséquilibrer son "
                           "ensemble de données d'entraînement, ou seulement revérifier les détections avec un niveau de confiance moyen. Après avoir sélectionné les images, "
                           "vous pourrez les vérifier. Après avoir vérifié toutes les images sélectionnées, vous serez invité à créer des données d'entraînement. Dans ce cas, "
                           "les images sélectionnées et leurs fichiers d'annotation associés recevront un nom unique et seront déplacés ou copiés dans le dossier de votre choix."
                           " Ceci est particulièrement pratique pour créer des données d'entraînement, car pour l'entraînement, tous les fichiers doivent se trouver dans un seul "
                           "dossier. Ainsi, les fichiers seront uniques et vous n'aurez pas de problèmes de remplacement lors de l'ajout de fichiers à vos données d'entraînement "
                           "existantes. Vous pouvez également ignorer les données d'entraînement et simplement continuer à post-traiter les résultats vérifiés. Non applicable "
                           "aux vidéos.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # forth step
    help_text_widget.insert(END, t('fth_step') + "\n")
    help_text_widget.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # destination folder
    help_text_widget.insert(END, f"{t('lbl_output_dir')}\n")
    help_text_widget.insert(END, ["Here you can browse for a folder in which the results of the post-processing features will be placed. If nothing is selected, the folder "
                    "chosen at step one will be used as the destination folder.\n\n",
                    "Aquí puede buscar una carpeta en la que se colocarán los resultados de las funciones de postprocesamiento. Si no se selecciona nada, la carpeta "
                    "elegida en el primer paso se utilizará como carpeta de destino.\n\n",
                    "Vous pouvez ici sélectionner le dossier dans lequel seront placés les résultats des fonctions de post-traitement. Si rien n'est sélectionné, le "
                    "dossier choisi à l'étape 1 sera utilisé comme dossier de destination.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # separate files
    help_text_widget.insert(END, f"{t('lbl_separate_files')}\n")
    help_text_widget.insert(END, ["This function divides the files into subdirectories based on their detections. Please be warned that this will be done automatically. "
                    "There will not be an option to review and adjust the detections before the images will be moved. If you want that (a human in the loop), take a look at ",
                    "Esta función divide los archivos en subdirectorios en función de sus detecciones. Tenga en cuenta que esto se hará automáticamente. No habrá opción de "
                    "revisar y ajustar las detecciones antes de mover las imágenes. Si quieres eso (una humano en el bucle), echa un vistazo a ",
                    "Cette fonction divise les fichiers en sous-répertoires en fonction de leurs détections. Veuillez noter que cette opération sera effectuée "
                    "automatiquement. Il n'y aura pas d'option permettant de vérifier et d'ajuster les détections avant le déplacement des images. Si vous souhaitez une "
                    "intervention humaine, consultez la section "][i18n_lang_idx()])
    help_text_widget.insert(INSERT, "Timelapse", hyperlink.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/")))
    help_text_widget.insert(END, [", which offers such a feature. More information about that ",
                           ", que ofrece tal característica. Más información al respecto ",
                           ", qui offre une telle fonctionnalité. Plus d'informations à ce sujet "][i18n_lang_idx()])
    help_text_widget.insert(INSERT, t('here'), hyperlink.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/uploads/Guides/TimelapseImageRecognitionGuide.pdf")))
    help_text_widget.insert(END,[" (starting on page 9). The process of importing the output file produced by AddaxAI into Timelapse is described ",
                          " (a partir de la página 9). El proceso de importación del archivo de salida producido por AddaxAI en Timelapse se describe ",
                          " (à partir de la page 9). Le processus d'importation du fichier de sortie produit par AddaxAI dans Timelapse est décrit "][i18n_lang_idx()])
    help_text_widget.insert(INSERT, t('here'), hyperlink.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/pmwiki.php?n=Main.DownloadMegadetector")))
    help_text_widget.insert(END,".\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # method of file placement
    help_text_widget.insert(END, f"{t('lbl_file_placement')}\n")
    help_text_widget.insert(END, ["Here you can choose whether to move the files into subdirectories, or copy them so that the originals remain untouched.\n\n",
                           "Aquí puedes elegir si quieres mover los archivos a subdirectorios o copiarlos de forma que los originales permanezcan intactos.\n\n",
                           "Ici, vous pouvez choisir de DÉPLACER les fichiers dans des sous-répertoires ou de les COPIER afin que les originaux restent intacts.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # sort results based on confidence
    help_text_widget.insert(END, f"{t('lbl_sep_conf')}\n")
    help_text_widget.insert(END, ["This feature will further separate the files based on its confidence value (in tenth decimal intervals). That means that each class will"
                        " have subdirectories like e.g. 'conf_0.6-0.7', 'conf_0.7-0.8', 'conf_0.8-0.9', etc.\n\n",
                        "Esta función separará aún más los archivos en función de su valor de confianza (en intervalos decimales). Esto significa que cada clase tendrá"
                        " subdirectorios como, por ejemplo, 'conf_0.6-0.7', 'conf_0.7-0.8', 'conf_0.8-0.9', etc.\n\n",
                        "Cette fonctionnalité permet de séparer les fichiers selon leur niveau de confiance (au dixième près). Cela signifie que chaque classe aura "
                        "des sous-répertoires tels que « conf_0.6-0.7 », « conf_0.7-0.8 », « conf_0.8-0.9 », etc.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # visualize files
    help_text_widget.insert(END, f"{t('lbl_vis_files')}\n")
    help_text_widget.insert(END, ["This functionality draws boxes around the detections and prints their confidence values. This can be useful to visually check the results."
                    " Videos can't be visualized using this tool. Please be aware that this action is permanent and cannot be undone. Be wary when using this on original images.\n\n",
                    "Esta funcionalidad dibuja recuadros alrededor de las detecciones e imprime sus valores de confianza. Esto puede ser útil para comprobar visualmente los "
                    "resultados. Los vídeos no pueden visualizarse con esta herramienta.\n\n",
                    "Cette fonctionnalité encadre les détections et affiche leurs scores de confiance. Cela peut être utile pour vérifier visuellement les résultats. Les vidéos "
                    "ne peuvent pas être visualisées avec cet outil. Veuillez noter que cette action est permanente et irréversible. Soyez prudent lorsque vous l'utilisez sur "
                    "des images originales.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # crop files
    help_text_widget.insert(END, f"{t('lbl_crp_files')}\n")
    help_text_widget.insert(END, ["This feature will crop the detections and save them as separate images. Not applicable for videos.\n\n",
                           "Esta función recortará las detecciones y las guardará como imágenes separadas. No es aplicable a los vídeos.\n\n",
                           "Cette fonctionnalité recadre les détections et les enregistre sous forme d'images distinctes. Non applicable aux vidéos.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # plot graphs
    help_text_widget.insert(END, f"{t('lbl_plt')}\n")
    help_text_widget.insert(END, ["Here you can select to create activity patterns, bar charts, pie charts and temporal heatmaps. The time unit (year, month, "
                           "week or day) will be chosen automatically based on the time period of your data. If more than 100 units are needed to "
                           "visualize, they will be skipped due to long processing times. Each visualization results in a static PNG file and a dynamic"
                           " HTML file to explore the data further. Additional interactive maps will be produced when geotags can be retrieved from the "
                           "image metadata.", "Aquí puede seleccionar la creación de patrones de actividad, gráficos de barras, gráficos circulares y "
                           "mapas térmicos temporales. La unidad temporal (año, mes, semana o día) se elegirá automáticamente en función del periodo de"
                           " tiempo de sus datos. Si se necesitan más de 100 unidades para visualizar, se omitirán debido a los largos tiempos de "
                           "procesamiento. Cada visualización da como resultado un archivo PNG estático y un archivo HTML dinámico para explorar más a "
                           "fondo los datos. Se producirán mapas interactivos adicionales cuando se puedan recuperar geoetiquetas de los metadatos de "
                           "las imágenes.",
                           "Vous pouvez ici créer des modèles d'activité, des graphiques à barres, des diagrammes circulaires et des cartes thermiques "
                           "temporelles. L'unité de temps (année, mois, semaine ou jour) sera automatiquement choisie en fonction de la période de vos "
                           "données. Si plus de 100 unités sont nécessaires à la visualisation, elles seront ignorées en raison des longs délais de "
                           "traitement. Chaque visualisation génère un fichier PNG statique et un fichier HTML dynamique pour une exploration plus "
                           "approfondie des données. Des cartes interactives supplémentaires seront générées lorsque les géotags pourront être "
                           "récupérés à partir des métadonnées de l'image."][i18n_lang_idx()])
    help_text_widget.insert(END, "\n\n")
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # export results
    help_text_widget.insert(END, f"{t('lbl_exp')}\n")
    help_text_widget.insert(END, ["Here you can select whether you want to export the results to other file formats. It will additionally try to fetch image metadata, like "
                           "timestamps, locations, and more.\n\n", "Aquí puede seleccionar si desea exportar los resultados a otros formatos de archivo. Además, "
                           "intentará obtener metadatos de la imagen, como marcas de tiempo, ubicaciones, etc. \n\n",
                           "Ici, vous pouvez choisir d'exporter les résultats vers d'autres formats de fichier. L'outil essaiera également de récupérer les "
                           "métadonnées de l'image, comme les horodatages, les emplacements, etc.\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # postprocess confidence threshold
    help_text_widget.insert(END, f"{t('lbl_thresh')}\n")
    help_text_widget.insert(END, ["Detections below this value will not be post-processed. To adjust the threshold value, you can drag the slider or press either sides next to "
                    "the slider for a 0.01 reduction or increment. Confidence values are within the [0.01, 1] interval. If you set the confidence threshold too high, "
                    "you will miss some detections. On the other hand, if you set the threshold too low, you will get false positives. When choosing a threshold for your "
                    f"project, it is important to choose a threshold based on your own data. My advice is to first visualize your data ('{t('lbl_vis_files')}') with a low "
                    "threshold to get a feeling of the confidence values in your data. This will show you how sure the model is about its detections and will give you an "
                    "insight into which threshold will work best for you. If you really don't know, 0.2 is probably a conservative threshold for most projects.\n\n",
                    "Las detecciones por debajo de este valor no se postprocesarán. Para ajustar el valor del umbral, puede arrastrar el control deslizante o pulsar "
                    "cualquiera de los lados junto al control deslizante para una reducción o incremento de 0,01. Los valores de confianza están dentro del intervalo "
                    "[0,01, 1]. Si ajusta el umbral de confianza demasiado alto, pasará por alto algunas detecciones. Por otro lado, si fija el umbral demasiado bajo, "
                    "obtendrá falsos positivos. Al elegir un umbral para su proyecto, es importante elegir un umbral basado en sus propios datos. Mi consejo es que primero"
                    f" visualice sus datos ('{t('lbl_vis_files')}') con un umbral bajo para hacerse una idea de los valores de confianza de sus datos. Esto le mostrará lo "
                    "seguro que está el modelo sobre sus detecciones y le dará una idea de qué umbral funcionará mejor para usted. Si realmente no lo sabe, 0,2 es "
                    "probablemente un umbral conservador para la mayoría de los proyectos.\n\n",
                    "Les détections inférieures à cette valeur ne seront pas traitées ultérieurement. Pour ajuster la valeur seuil, faites glisser le curseur ou appuyez sur l'un "
                    "des côtés de la barre de défilement par incrément ou décrément de 0.01. Les valeurs de confiance sont comprises entre [0,01 et 1]. Si le seuil de confiance "
                    "est trop élevé, vous manquerez certaines détections. En revanche, si vous définissez un seuil trop bas, vous obtiendrez des faux positifs. Lorsque vous "
                    "choisissez un seuil pour votre projet, il est important de choisir un seuil en fonction de vos propres données. Je vous conseille de commencer par visualiser "
                    f"vos données ('{t('lbl_vis_files')}') avec un seuil bas pour obtenir une idée du niveau de confiance dans vos données. Cela vous montrera à quel "
                    "point le modèle est sûr de ses détections et vous donnera une intuition du seuil le plus adapté à vos besoins. Si vous ne le savez pas vraiment, 0,2 est "
                    "probablement un seuil prudent pour la plupart des projets..\n\n"][i18n_lang_idx()])
    help_text_widget.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text_widget.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # config help_text
    help_text_widget.pack(fill="both", expand=True)
    help_text_widget.configure(font=(text_font, 11, "bold"), state=DISABLED)
    if scroll is not None:
        scroll.configure(command=help_text_widget.yview)
