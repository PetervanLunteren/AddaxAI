# Frontend Conventions

This document outlines the conventions, patterns, and design decisions for the AddaxAI frontend.

## Tech Stack

- **Framework**: React 19 with TypeScript (strict mode)
- **Build Tool**: Vite 7
- **Routing**: React Router DOM v6
- **State Management**:
  - TanStack Query (React Query) for server state
  - Zustand for client state (when needed)
- **Forms**: React Hook Form + Zod validation
- **Styling**: Tailwind CSS v3
- **UI Components**: shadcn/ui (Radix UI primitives)

## Design System

### Colors

We use a HSL-based color system defined in `src/index.css`:

```css
--background: 0 0% 100%
--foreground: 222.2 84% 4.9%
--primary: #0f6064              /* Teal - Main brand color */
--secondary: #ebf0f2            /* Light teal - Supporting elements */
--card-background: #f5f7fb      /* Grey - Card/container backgrounds */
--accent: 210 40% 96.1%
--destructive: 0 84.2% 60.2%
--muted: 210 40% 96.1%
--border: 214.3 31.8% 91.4%
```

**Color Usage:**
- **Primary (#0f6064)**: Main actions, buttons, important elements, checkboxes, brand elements
- **Secondary (#e4ecee)**: Supporting actions, subtle backgrounds, hover states
- **Card Background (#f5f7fb)**: Cards, containers, panels, background sections
- **Destructive**: Delete, danger actions
- **Muted**: Disabled states, subtle text

### Typography

- Font: System font stack (default Tailwind)
- Headings: `text-2xl font-bold tracking-tight`
- Body: Default size (16px/1rem)
- Small text: `text-sm` or `text-xs`
- Muted text: `text-muted-foreground`

### Spacing

Follow Tailwind's spacing scale:
- Card padding: `p-6`
- Form spacing: `space-y-4`
- Grid gaps: `gap-6`
- Page padding: `px-4 py-8 sm:px-6 lg:px-8`

### Layout Patterns

**Container Width:**
```tsx
<div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
```

**Card Grid:**
```tsx
<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
```

**Gradient Background:**
```tsx
<div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
```

**Header with Backdrop Blur:**
```tsx
<header className="border-b bg-white/80 backdrop-blur-sm">
```

## Icon Library

**Library**: [Lucide React](https://lucide.dev/)

**Installation**: `lucide-react` (already installed)

**Common Icons Used:**
- `Camera`: Main app icon
- `Plus`: Create/add actions
- `Settings`: Edit/configure actions
- `ArrowLeft`: Back navigation
- `FolderTree`: Project/folder structures
- `BarChart3`: Analytics/stats
- `Upload`: File upload
- `Download`: Export/download

**Usage Pattern:**
```tsx
import { Camera, Plus, Settings } from "lucide-react";

<Settings className="h-4 w-4" />  // Small icons (buttons)
<Camera className="h-6 w-6" />     // Medium icons (headers)
```

## Component Patterns

### shadcn/ui Components

Location: `src/components/ui/`

**Available Components:**
- `Button`: Primary UI actions
- `Card`: Content containers
- `Dialog`: Modals/popups
- `Form`: Form wrapper with context
- `Input`: Text inputs
- `Label`: Form labels
- `Textarea`: Multi-line text (when needed)

**Adding New Components:**
Use the shadcn CLI pattern - copy from [ui.shadcn.com](https://ui.shadcn.com) and adapt.

### Form Pattern

Standard pattern for forms using React Hook Form + Zod:

```tsx
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";

const schema = z.object({
  name: z.string().min(1, "Name is required"),
  optional_field: z
    .string()
    .optional()
    .transform((val) => (val === "" ? undefined : val)),
});

const form = useForm({
  resolver: zodResolver(schema),
  defaultValues: { name: "", optional_field: "" },
});
```

**Important**: Transform empty strings to `undefined` for optional fields to match backend expectations.

### Dialog Pattern

```tsx
const [open, setOpen] = useState(false);

<Dialog open={open} onOpenChange={setOpen}>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Title</DialogTitle>
      <DialogDescription>Description</DialogDescription>
    </DialogHeader>
    {/* Content */}
    <DialogFooter>
      <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
      <Button type="submit">Confirm</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
```

### API Query Pattern

Using TanStack Query for data fetching:

```tsx
const { data, isLoading } = useQuery({
  queryKey: ["resource", id],
  queryFn: () => api.get(id),
  enabled: !!id,  // Only run if id exists
});

const mutation = useMutation({
  mutationFn: (data) => api.create(data),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ["resource"] });
    onClose();
  },
  onError: (error: Error) => {
    form.setError("root", { message: error.message });
  },
});
```

## File Structure

```
frontend/
├── src/
│   ├── api/              # API client and type definitions
│   │   ├── projects.ts
│   │   ├── sites.ts
│   │   └── types.ts
│   ├── components/
│   │   ├── ui/           # shadcn/ui primitives
│   │   ├── projects/     # Project-specific components
│   │   └── sites/        # Site-specific components
│   ├── pages/            # Route pages
│   │   ├── ProjectsPage.tsx
│   │   └── ProjectDetailPage.tsx
│   ├── lib/              # Utilities and config
│   │   ├── api-client.ts
│   │   ├── query-client.ts
│   │   └── utils.ts
│   ├── App.tsx           # Main app with routes
│   ├── index.css         # Global styles
│   └── main.tsx          # Entry point
```

## Navigation Structure (Planned)

### Current Routes
- `/` → Redirect to `/projects`
- `/projects` → Projects list
- `/projects/:projectId` → Project detail with sites

### Future Routes (To Be Implemented)
- `/projects/:projectId/sites/:siteId` → Site detail with deployments
- `/projects/:projectId/files` → File browser/uploader
- `/projects/:projectId/detections` → Detection results
- `/settings` → App settings

### Sidebar Navigation (To Be Implemented)

**Layout**: App will have a persistent sidebar with:
- Logo/branding at top
- Navigation links
- Project switcher
- Settings at bottom

**Structure**:
```
┌─────────────┬──────────────────┐
│  Sidebar    │   Main Content   │
│             │                  │
│ 🏠 Projects │                  │
│ 📁 Files    │                  │
│ 🔍 Search   │                  │
│ 📊 Reports  │                  │
│             │                  │
│ ⚙️ Settings │                  │
└─────────────┴──────────────────┘
```

## Styling Conventions

### Tailwind Utilities

**Prefer utility classes over custom CSS.** Only create custom CSS for truly reusable patterns.

**Common Patterns:**
```tsx
// Hover effects
className="transition-shadow hover:shadow-lg"
className="hover:bg-accent hover:text-accent-foreground"

// Focus states (automatically handled by shadcn components)
className="focus:outline-none focus:ring-2 focus:ring-ring"

// Responsive design
className="grid gap-6 md:grid-cols-2 lg:grid-cols-3"
className="px-4 sm:px-6 lg:px-8"
```

### Component Composition

Prefer composition over configuration:

```tsx
// Good
<Button variant="destructive" size="sm">Delete</Button>

// Also good
<Card className="transition-shadow hover:shadow-lg">
  <CardHeader>...</CardHeader>
  <CardContent>...</CardContent>
</Card>
```

## Type Safety

### Import Types Separately

For type-only imports to avoid runtime issues:

```tsx
import { useForm } from "react-hook-form";
import type { FieldValues } from "react-hook-form";  // Type-only import
```

### API Types

All API types defined in `src/api/types.ts` matching backend Pydantic schemas:

```tsx
export interface ProjectCreate {
  name: string;
  description?: string | null;
}

export interface ProjectResponse {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}
```

## Error Handling

### Form Errors

```tsx
onError: (error: Error) => {
  form.setError("root", { message: error.message });
}

// Display in form
{form.formState.errors.root && (
  <p className="text-sm font-medium text-destructive">
    {form.formState.errors.root.message}
  </p>
)}
```

### Delete Confirmations

Always confirm destructive actions:

```tsx
onClick={() => {
  if (confirm("Are you sure? This action cannot be undone.")) {
    deleteMutation.mutate();
  }
}}
```

## Performance Considerations

- Use `React.memo()` sparingly - only for expensive components
- Leverage TanStack Query's caching - don't over-invalidate
- Use proper `queryKey` patterns for cache granularity
- Avoid inline function definitions in render when possible

## Accessibility

- All interactive elements should be keyboard accessible
- Use semantic HTML (`<button>`, `<nav>`, etc.)
- shadcn/ui components include ARIA attributes
- Ensure proper focus management in dialogs

## Code Style

- Use functional components with hooks
- Prefer `const` over `let`
- Use arrow functions for components
- Follow the pattern: imports → component → exports
- Add JSDoc comments for complex logic
- Keep components single-purpose

### Text Capitalization

**No Title Case** - Use natural English capitalization throughout the UI:
- Only capitalize the first word of sentences and proper nouns
- Examples:
  - ✅ "Create new project", "Edit project", "Select species"
  - ✅ "Project name", "Detection model", "Species taxonomy"
  - ✅ "Cities visited: Utrecht, Amsterdam"
  - ❌ "Create New Project", "Edit Project", "Select Species"
  - ❌ "Project Name", "Detection Model", "Species Taxonomy"
- Proper nouns remain capitalized: "MegaDetector", "SpeciesNet", "Peter van Lunteren"

## Testing (To Be Implemented)

Plan to add:
- Vitest for unit tests
- Testing Library for component tests
- Playwright for E2E tests

## Future Enhancements

1. **Dark Mode**: Add theme switcher using Tailwind dark mode
2. **Internationalization**: Add i18n support if needed
3. **Animations**: Consider framer-motion for complex animations
4. **Data Visualization**: Add chart library (recharts or visx)
5. **Image Gallery**: Component for viewing camera trap images
6. **Map View**: Leaflet integration for site locations
