# Developer Documentation

**Key principles:**
1. **Crash early and loudly** - Fail hard in development so bugs cannot hide. Never allow silent failures.
2. **Explicit configuration** - No defaults. If something is missing, stop and surface it immediately.
3. **Type hints everywhere** - Make expectations clear and support safe refactoring.
4. **Short and clear documentation** - Keep explanations concise without losing clarity.
5. **Open source friendly** - Never commit secrets or anything that should not be public.
6. **No backward compatibility** - The project is in motion and has no users. Refactor freely when needed.
7. **Prefer simple solutions** - Use straightforward approaches that follow the conventions. Avoid cleverness when simplicity works.
8. **Follow the established conventions** - Keep structure predictable so the codebase stays readable and easy to maintain. 
9. **No quick fixes** - Fix issues in a way that holds for all future deployments, not only the current device.
10. **GitHub:** Always commit manually. Never commit automatically. 

**Remember:** It's better to crash during development than to hide bugs that cause problems later. We'll add resilience (retries, fallbacks, graceful degradation) after the core functionality works.

