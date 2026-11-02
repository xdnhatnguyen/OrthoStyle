// Edit only this file when you want to change the demo gallery.
// IDs should match the filenames in assets/content and assets/style.

window.PROJECT_DEMO = {
  content: [
    {
      id: "01_backpack_dog",
      label: "Backpack",
      src: "assets/content/01_backpack_dog.webp"
    },
    {
      id: "02_bear_plushie",
      label: "Bear",
      src: "assets/content/02_bear_plushie.webp"
    },
    {
      id: "03_berry_bowl",
      label: "Berry",
      src: "assets/content/03_berry_bowl.webp"
    },
    {
      id: "04_can",
      label: "Can",
      src: "assets/content/04_can.webp"
    },
    {
      id: "05_cat_laying",
      label: "Cat",
      src: "assets/content/05_cat_laying.webp"
    },
    {
      id: "06_cat_sitting",
      label: "Cat",
      src: "assets/content/06_cat_sitting.webp"
    },
    {
      id: "07_vintage_clock",
      label: "Clock",
      src: "assets/content/07_vintage_clock.webp"
    },
    {
      id: "08_colorful_sneaker",
      label: "Sneaker",
      src: "assets/content/08_colorful_sneaker.webp"
    },
    {
      id: "09_dog_sitting",
      label: "Dog",
      src: "assets/content/09_dog_sitting.webp"
    },
    {
      id: "10_rubber_duck",
      label: "Duck",
      src: "assets/content/10_rubber_duck.webp"
    },
    {
      id: "11_fancy_boot",
      label: "Boot",
      src: "assets/content/11_fancy_boot.webp"
    },
    {
      id: "12_pink_sunglasses",
      label: "Sunglasses",
      src: "assets/content/12_pink_sunglasses.webp"
    },
    {
      id: "13_poop_emoji",
      label: "Poop Emoji",
      src: "assets/content/13_poop_emoji.webp"
    },
    {
      id: "14_ceramic_teapot",
      label: "Teapot",
      src: "assets/content/14_ceramic_teapot.webp"
    },
    {
      id: "15_decorative_vase",
      label: "Vase",
      src: "assets/content/15_decorative_vase.webp"
    }
  ],
  style: [
    {
      id: "01_antimonocromatismo",
      label: "Antimonocromatismo",
      src: "assets/style/01_antimonocromatismo.webp"
    },
    {
      id: "02_color_splash",
      label: "Color Splash",
      src: "assets/style/02_color_splash.webp"
    },
    {
      id: "03_cyberpunk",
      label: "Cyberpunk",
      src: "assets/style/03_cyberpunk.webp"
    },
    {
      id: "04_flat_illustration",
      label: "Flat",
      src: "assets/style/04_flat_illustration.webp"
    },
    {
      id: "05_glowing_neon",
      label: "Glowing",
      src: "assets/style/05_glowing_neon.webp"
    },
    {
      id: "06_historical_oil",
      label: "Oil Painting",
      src: "assets/style/06_historical_oil.webp"
    },
    {
      id: "07_graphic_poster",
      label: "Poster",
      src: "assets/style/07_graphic_poster.webp"
    },
    {
      id: "08_pencil_sketch",
      label: "Pencil Sketch",
      src: "assets/style/08_pencil_sketch.webp"
    },
    {
      id: "09_sticker",
      label: "Sticker",
      src: "assets/style/09_sticker.webp"
    },
    {
      id: "10_starry_night_sketch",
      label: "Starry Night",
      src: "assets/style/10_starry_night_sketch.webp"
    },
    {
      id: "11_abstract_3d_render",
      label: "3D Render",
      src: "assets/style/11_abstract_3d_render.webp"
    },
    {
      id: "12_vintage_woodcut",
      label: "Woodcut",
      src: "assets/style/12_vintage_woodcut.webp"
    },
    {
      id: "13_3d_isometric",
      label: "Isometric",
      src: "assets/style/13_3d_isometric.webp"
    },
    {
      id: "14_crayon_drawing",
      label: "Crayon",
      src: "assets/style/14_crayon_drawing.webp"
    },
    {
      id: "15_oil_pastels",
      label: "Oil Pastels",
      src: "assets/style/15_oil_pastels.webp"
    }
  ],

  // The loader tries each extension in this order.
  // For your real project, place files like:
  // assets/results/backpack__flat.webp
  resultExtensions: ["webp", "png", "jpg", "jpeg", "svg"],
  resultDirectory: "assets/results",

  // Purely cosmetic: gives the button a short "generation" feel.
  // No model is executed; the result is a static image.
  fakeLoadingMs: 1000
};
