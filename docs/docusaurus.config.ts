import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import { themes as prismThemes } from "prism-react-renderer";

// AddaxAI documentation + landing site.
//
// Layout: a custom marketing homepage at "/" (src/pages/index.tsx) and the
// documentation under "/docs". Deployed to GitHub Pages at
// https://petervanlunteren.github.io/AddaxAI-WebUI/ by
// .github/workflows/docs.yml on every push to main.
//
// If you later point a custom domain (e.g. docs.addaxdatascience.com) at the
// site, change `url` to that domain and `baseUrl` to "/", and drop a CNAME
// file into static/.
const config: Config = {
  title: "AddaxAI",
  tagline: "Simplifying camera trap image analysis with AI",
  favicon: "img/favicon.ico",

  future: {
    v4: true,
  },

  url: "https://petervanlunteren.github.io",
  baseUrl: "/AddaxAI-WebUI/",

  organizationName: "PetervanLunteren",
  projectName: "AddaxAI-WebUI",

  // 'warn' keeps the scaffold building while pages are still being written.
  // Tighten to 'throw' once the content settles so dead links fail the build.
  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  markdown: {
    mermaid: true,
  },
  themes: [
    "@docusaurus/theme-mermaid",
    [
      // Offline, account-free search. Indexes the built docs at deploy time.
      "@easyops-cn/docusaurus-search-local",
      {
        hashed: true,
        indexBlog: false,
        docsRouteBasePath: "/docs",
        highlightSearchTermsOnTargetPage: true,
      },
    ],
  ],

  presets: [
    [
      "classic",
      {
        docs: {
          routeBasePath: "/docs",
          sidebarPath: "./sidebars.ts",
          editUrl:
            "https://github.com/PetervanLunteren/AddaxAI-WebUI/tree/main/docs/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: "img/logo-wordmark.png",
    colorMode: {
      defaultMode: "light",
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "AddaxAI",
      logo: {
        alt: "AddaxAI",
        src: "img/logo-mark.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "docs",
          position: "left",
          label: "Documentation",
        },
        { to: "/docs/faq", label: "FAQ", position: "left" },
        {
          href: "https://github.com/PetervanLunteren/AddaxAI-WebUI/releases/latest",
          label: "Download",
          position: "right",
        },
        {
          href: "https://github.com/PetervanLunteren/AddaxAI-WebUI",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            { label: "Getting started", to: "/docs/getting-started/installation" },
            { label: "Model zoo", to: "/docs/models/model-zoo" },
            { label: "FAQ", to: "/docs/faq" },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/PetervanLunteren/AddaxAI-WebUI",
            },
            {
              label: "Addax Data Science",
              href: "https://addaxdatascience.com",
            },
            {
              label: "Contact",
              href: "mailto:peter@addaxdatascience.com",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Addax Data Science. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
