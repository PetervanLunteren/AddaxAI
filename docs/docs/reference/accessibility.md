---
sidebar_position: 5
title: Accessibility
---

# Accessibility

What works if you use a keyboard, a screen reader or a magnifier, what does not work yet, and where to get the formal report your organisation may ask for.

## The conformance report

Some organisations, for example US federal agencies, need an Accessibility Conformance Report (ACR, often called a VPAT) before they can approve software. AddaxAI has one:

[Download the Accessibility Conformance Report (Word document, 70 KB)](https://github.com/PetervanLunteren/AddaxAI/releases/download/v7.7.0/AddaxAI-accessibility-conformance-report-2026-09.docx)

It follows the ITI VPAT 2.5 Revised Section 508 edition. It covers WCAG 2.0 Level A and AA and the Revised Section 508 standards, for AddaxAI 7.7.0 and this website, as evaluated in September 2026.

I wrote it myself. It is not an independent audit. It says where AddaxAI falls short as plainly as where it does well.

## What works

- The application menu is the operating system's own menu. It works with every screen reader and opens with the standard keys: F10 or Alt on Windows and Linux, Ctrl+F2 on macOS.
- Zoom. Use View, then Zoom in, Zoom out or Reset zoom, or the usual shortcuts (Ctrl or Cmd with plus, minus and 0). Nothing in the app blocks zoom, and the display scaling of your operating system is respected.
- Dialogs, forms, settings and tables work with the keyboard. Tab moves between controls, Enter and Space activate them, Escape closes a dialog and puts focus back where it was. Every control shows a focus ring.
- Keyboard shortcuts on the labels page. The common review actions (verify, reject, relabel, next) are single keys. The list is behind the keyboard button in the toolbar, and you can give the keys 1 to 9 your own species.
- Colour is never the only cue for a status. Verified, favourite and flagged badges have different shapes, species chips carry the name, and messages carry an icon.
- Screen readers get through, because the app runs on Chromium, which talks to VoiceOver, Narrator, NVDA, JAWS and Orca.

## What does not work yet

AddaxAI is a tool for looking at photos and deciding which animal is in them, and the review screens are built for a mouse. Know these limits before you plan to use it with a keyboard alone or with a screen reader:

- The labels and counts grids cannot be reached by keyboard. The tiles and cards are not focusable, so the review itself needs a mouse.
- Bounding boxes are drawn, moved and resized with the mouse only.
- In the event viewer, Enter jumps to the next event and Space starts or stops the slideshow. So the buttons inside it cannot be pressed from the keyboard.
- Many controls have no spoken name. A screen reader announces the filter dropdowns and a number of icon-only buttons as unnamed.
- Charts and the map have no text alternative. The numbers behind them are in the exports.
- Some colours are below the recommended contrast. Mainly the delete buttons, the numbers inside the confusion matrix and a few small hints.
- There is no dark or high contrast theme. Windows high contrast mode is applied by the browser engine, but the app does not follow the system colour scheme.
- The tutorial videos on this site have automatic captions only.

The full list, criterion by criterion, is in the report.

## Report a problem

If you hit an accessibility barrier, email me at [peter@addaxdatascience.com](mailto:peter@addaxdatascience.com) or [open an issue on GitHub](https://github.com/PetervanLunteren/AddaxAI/issues). Say which screen, which assistive technology and what happened. Support is by email, so it works with any text-based tool.
