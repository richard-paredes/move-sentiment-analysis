"use client"

import { z } from "zod"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useTheme } from "next-themes";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { DropdownMenuTrigger } from "@radix-ui/react-dropdown-menu";
import { MoonIcon, SunIcon } from "@radix-ui/react-icons";

const formSchema = z.object({
  movieReview: z.string().min(5, {
    message: "Text to analyze must be at least 5 characters."
  })
})

export function NavigationMenu() {
  const { setTheme } = useTheme();
  const themes = ["light", "dark", "system"];
  function ThemeItem({ theme }: { theme: string }) {
    return <DropdownMenuItem onClick={() => setTheme(theme)}>
      {theme.charAt(0).toUpperCase() + theme.slice(1)}
    </DropdownMenuItem>
  }

  return <nav className="flex min-w-full p-2">
    <div>
      RP
    </div>
    <div className="ml-auto mx-4">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="icon">
            <SunIcon className="h-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <MoonIcon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {themes.map(t => <ThemeItem theme={t} />)}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  </nav>
}

export default function Home() {

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      movieReview: ""
    },
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    alert(`You submitted ${JSON.stringify(values)}`);
  }

  return (
    <main className="flex min-h-screen flex-col items-center">
      <NavigationMenu />
      <section className="p-24">
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
            <FormField control={form.control} name="movieReview" render={({ field }) => {
              return <FormItem>
                <FormLabel>Message</FormLabel>
                <FormControl>
                  <Input placeholder="That movie was fantastic!" {...field} />
                </FormControl>
                <FormDescription>
                  Receive sentiment analysis for the provided message.
                </FormDescription>
              </FormItem>
            }} />
            <Button type="submit">Submit</Button>
          </form>
        </Form>
      </section>
    </main>
  );
}
